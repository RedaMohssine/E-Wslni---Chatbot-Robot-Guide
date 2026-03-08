"""
Terminal-based RAG chatbot with text and voice modes.
Uses Gemini for generation, Whisper for STT, Edge TTS for speech.
"""

import os
import re
import warnings
import asyncio
import tempfile

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb
import google.generativeai as genai_old
import whisper
import edge_tts
from pygame import mixer

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "emines_rag"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
TOP_K = 5

SYSTEM_PROMPT = """Tu es un assistant intelligent et amical intégré dans un robot autonome 
qui guide les visiteurs et étudiants sur le campus de l'Université Mohammed VI 
Polytechnique (UM6P), en particulier à l'EMINES - School of Industrial Management.

TES REGLES :
1. Réponds UNIQUEMENT à partir du contexte fourni ci-dessous. Ne fabrique JAMAIS d'informations.
2. Si la réponse n'est pas dans le contexte, dis poliment : "Je n'ai pas cette information 
   dans ma base de données. Je vous suggère de contacter l'EMINES directement à contact@eminesingenieur.org"
3. Réponds dans la MEME LANGUE que la question (français, anglais, arabe...).
4. Sois concis mais complet. Utilise des listes à puces quand c'est pertinent.
5. Sois chaleureux et accueillant — tu représentes l'université !
6. Si on te pose une question sur la localisation, mentionne que le campus est à Ben Guérir, 
   à 40 minutes au nord de Marrakech.
7. NE répète PAS de formules de politesse comme "Je suis ravi de vous aider" à chaque réponse. 
   Réponds directement à la question posée.

CONTEXTE (extraits de la base de connaissances EMINES/UM6P) :
---
{context}
---

Réponds à la question suivante de manière claire et utile :"""

WELCOME_MESSAGE = ("Bonjour ! Je suis votre assistant virtuel pour l'EMINES et l'UM6P. "
                   "N'hésitez pas à me poser vos questions sur les programmes, "
                   "l'admission, le campus, ou toute autre information !")

ABBREVIATIONS = {
    r'\bPr\b': 'Professeur', r'\bPr\.\b': 'Professeur',
    r'\bDr\b': 'Docteur', r'\bDr\.\b': 'Docteur',
    r'\bM\.\b': 'Monsieur', r'\bMme\b': 'Madame', r'\bMlle\b': 'Mademoiselle',
    r'\bSt\b': 'Saint', r'\bSte\b': 'Sainte',
    r'\bUniv\.\b': 'Université', r'\bFac\.\b': 'Faculté', r'\bIng\.\b': 'Ingénieur',
    r'\bNb\b': 'Nombre', r'\bNbr\b': 'Nombre',
    r'\bEx\b': 'Exemple', r'\bex\b': 'exemple',
    r'\betc\.\b': 'et cetera',
    r'\bTel\b': 'Téléphone', r'\bTél\b': 'Téléphone',
    r'\bAdm\.\b': 'Administration', r'\bResp\.\b': 'Responsable',
    r'\bDép\.\b': 'Département',
    r'\bBAC\b': 'Baccalauréat', r'\bBac\b': 'Baccalauréat',
    r'\bUM6P\b': 'U M 6 P', r'\bEMINES\b': 'EMINES',
    r'\bCPGE\b': 'classes préparatoires aux grandes écoles',
    r'\bTP\b': 'travaux pratiques', r'\bTD\b': 'travaux dirigés',
    r'\bCM\b': 'cours magistraux', r'\bECTS\b': 'crédits ECTS',
    r'\bLMD\b': 'licence master doctorat',
    r'\bR&D\b': 'recherche et développement',
}


# ---- Core RAG ----

def setup():
    """Initialize APIs and load the vector database."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
        print("ERROR: Please set GOOGLE_API_KEY in the .env file.")
        exit(1)
    client = genai.Client(api_key=api_key)
    genai_old.configure(api_key=api_key)
    db_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = db_client.get_collection(COLLECTION_NAME)
    print(f"API ready | {collection.count()} chunks loaded from ChromaDB")
    return client, collection


def retrieve(collection, question, top_k=TOP_K):
    """Embed the question and retrieve the most relevant chunks."""
    query_result = genai_old.embed_content(
        model=f"models/{EMBEDDING_MODEL}",
        content=question,
        task_type="retrieval_query",
    )
    results = collection.query(
        query_embeddings=[query_result['embedding']],
        n_results=top_k,
    )
    context_parts, sources = [], set()
    for i in range(len(results["documents"][0])):
        context_parts.append(results["documents"][0][i])
        sources.add(results["metadatas"][0][i].get("source", "unknown"))
    return "\n\n---\n\n".join(context_parts), sources


def generate(client, question, context, conversation_history=None):
    """Build augmented prompt and generate answer with Gemini."""
    full_prompt = SYSTEM_PROMPT.format(context=context)
    conversation_text = ""
    if conversation_history:
        conversation_text = "\n\nHISTORIQUE DE LA CONVERSATION :\n"
        for q, a in conversation_history[-5:]:
            conversation_text += f"\nUtilisateur: {q}\nAssistant: {a}\n"
        conversation_text += "\n---\n"
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=f"{full_prompt}{conversation_text}\n\n{question}",
        config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=4096),
    )
    return response.text


def ask(client, collection, question, conversation_history=None):
    """Full RAG pipeline: retrieve + generate."""
    context, sources = retrieve(collection, question)
    answer = generate(client, question, context, conversation_history)
    return answer, sources


# ---- Speech ----

WHISPER_MODEL = None


def init_whisper():
    """Load Whisper model (once)."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("Loading Whisper model...")
        WHISPER_MODEL = whisper.load_model("base")
        print("Whisper ready.")
    return WHISPER_MODEL


def clean_text_for_speech(text):
    """Strip markdown and expand abbreviations for TTS."""
    text = re.sub(r'\*{1,3}', '', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'`{1,3}', '', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^\s*[-•]\s+', '... ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '... ', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', ' ... ... ', text)
    text = re.sub(r'\n', ' ... ', text)
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def speak_async(text, voice="fr-FR-DeniseNeural"):
    """Generate speech with Edge TTS and play it."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tmp_path = tmp.name
    tmp.close()
    try:
        await edge_tts.Communicate(text, voice).save(tmp_path)
        mixer.init()
        mixer.music.load(tmp_path)
        mixer.music.play()
        while mixer.music.get_busy():
            await asyncio.sleep(0.1)
        mixer.music.unload()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def speak(text, voice="fr-FR-DeniseNeural"):
    """Synchronous TTS wrapper — cleans text then speaks."""
    asyncio.run(speak_async(clean_text_for_speech(text), voice))


def listen_whisper():
    """Record 5s of audio and transcribe with Whisper."""
    import sounddevice as sd
    import numpy as np

    model = init_whisper()
    duration = 5
    sample_rate = 16000

    print("Listening... (5 seconds)")
    try:
        recording = sd.rec(int(duration * sample_rate),
                           samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()
        print("Processing...")
        audio = recording.flatten()
        result = model.transcribe(audio, language=None, fp16=False)
        text = result["text"].strip()
        return text if text else None
    except Exception as e:
        print(f"Recording error: {e}")
        return None


# ---- Chat loops ----

def voice_chat_loop(client, collection):
    """Voice-enabled interactive chat loop."""
    print("\nVoice chatbot ready. Say 'quitter' or 'exit' to stop.\n")
    init_whisper()
    print(f"Assistant: {WELCOME_MESSAGE}\n")
    speak(WELCOME_MESSAGE)

    conversation_history = []
    while True:
        try:
            question = listen_whisper()
            if question is None:
                continue
            print(f"You: {question}\n")
            if any(w in question.lower() for w in ["quitter", "exit", "arrêter", "stop", "au revoir"]):
                goodbye = "Au revoir ! Bonne visite à l'UM6P !"
                print(f"Assistant: {goodbye}")
                speak(goodbye)
                break
            answer, sources = ask(client, collection, question, conversation_history)
            conversation_history.append((question, answer))
            print(f"Assistant:\n{answer}\n")
            speak(answer)
            if any(kw in question.lower() for kw in ['source', 'sources', 'référence', 'références']):
                print(f"Sources: {', '.join(list(sources)[:3])}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            error_msg = "Désolé, une erreur s'est produite. Pouvez-vous répéter?"
            print(f"Error: {e}")
            speak(error_msg)


def chat_loop(client, collection):
    """Text-based interactive chat loop."""
    print("\nText chatbot ready. Type 'quit' to exit.\n")
    print(f"Assistant: {WELCOME_MESSAGE}\n")

    conversation_history = []
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ["quit", "exit", "q"]:
            print("\nAu revoir !")
            break
        try:
            answer, sources = ask(client, collection, question, conversation_history)
            conversation_history.append((question, answer))
            print(f"\nAssistant:\n{answer}\n")
            if any(kw in question.lower() for kw in ['source', 'sources', 'référence', 'références']):
                print(f"Sources: {', '.join(list(sources)[:3])}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    client, collection = setup()
    print("\n1. Text mode\n2. Voice mode")
    choice = input("Choose (1 or 2): ").strip()
    if choice == "2":
        voice_chat_loop(client, collection)
    else:
        chat_loop(client, collection)
