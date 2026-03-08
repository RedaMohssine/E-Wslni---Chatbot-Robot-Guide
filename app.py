"""
E-Wslni — EMINES / UM6P RAG Chatbot
Streamlit UI with text + voice hybrid mode.
"""

import os
import re
import warnings
import asyncio
import tempfile

warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb
import google.generativeai as genai_old
import edge_tts

# --------------- CONFIG ---------------
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "emines_rag"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
TOP_K = 5
TTS_VOICE = "fr-FR-DeniseNeural"

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


ABBREVIATIONS = {
    r'\bPr\b': 'Professeur', r'\bPr\.\b': 'Professeur',
    r'\bDr\b': 'Docteur', r'\bDr\.\b': 'Docteur',
    r'\bM\.\b': 'Monsieur', r'\bMme\b': 'Madame', r'\bMlle\b': 'Mademoiselle',
    r'\bUniv\.\b': 'Université', r'\bFac\.\b': 'Faculté', r'\bIng\.\b': 'Ingénieur',
    r'\betc\.\b': 'et cetera',
    r'\bBAC\b': 'Baccalauréat', r'\bBac\b': 'Baccalauréat',
    r'\bUM6P\b': 'U M 6 P', r'\bEMINES\b': 'EMINES',
    r'\bCPGE\b': 'classes préparatoires aux grandes écoles',
    r'\bTP\b': 'travaux pratiques', r'\bTD\b': 'travaux dirigés',
    r'\bCM\b': 'cours magistraux', r'\bECTS\b': 'crédits ECTS',
    r'\bR&D\b': 'recherche et développement',
}


def clean_text_for_speech(text):
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


# --------------- CORE RAG ---------------
@st.cache_resource
def init_backend():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY in .env file")
        st.stop()
    client = genai.Client(api_key=api_key)
    genai_old.configure(api_key=api_key)
    db_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = db_client.get_collection(COLLECTION_NAME)
    return client, collection


def retrieve(collection, question, top_k=TOP_K):
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


def generate(client, question, context, history=None):
    full_prompt = SYSTEM_PROMPT.format(context=context)
    conv = ""
    if history:
        conv = "\n\nHISTORIQUE DE LA CONVERSATION :\n"
        for q, a in history[-5:]:
            conv += f"\nUtilisateur: {q}\nAssistant: {a}\n"
        conv += "\n---\n"
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=f"{full_prompt}{conv}\n\n{question}",
        config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=4096),
    )
    return response.text


def ask(client, collection, question, history=None):
    context, sources = retrieve(collection, question)
    return generate(client, question, context, history), sources


# --------------- TTS ---------------
async def _tts_to_bytes(text, voice=TTS_VOICE):
    cleaned = clean_text_for_speech(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()
    try:
        await edge_tts.Communicate(cleaned, voice).save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def tts_bytes(text, voice=TTS_VOICE):
    return asyncio.run(_tts_to_bytes(text, voice))


# --------------- STT ---------------
@st.cache_resource
def load_whisper():
    import whisper
    return whisper.load_model("base")


def transcribe_audio(audio_bytes):
    import numpy as np
    import wave

    model = load_whisper()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.write(audio_bytes)
    tmp.close()
    try:
        with wave.open(tmp_path, 'rb') as wf:
            n_ch = wf.getnchannels()
            fr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            audio_np = audio_np[::n_ch]
        if fr != 16000:
            from scipy.signal import resample
            audio_np = resample(audio_np, int(len(audio_np) * 16000 / fr)).astype(np.float32)
        return model.transcribe(audio_np, language=None, fp16=False)["text"].strip()
    except Exception:
        try:
            return model.transcribe(tmp_path, language=None, fp16=False)["text"].strip()
        except Exception:
            return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# =====================================================================
#  PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="E-Wslni | EMINES Assistant",
    page_icon="mortarboard",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =====================================================================
#  CSS
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #0088b0;
    --primary-light: #00a0c0;
    --accent: #d04020;
    --bg: #f0f4f6;
    --surface: #ffffff;
    --text: #1e293b;
    --text-muted: #64748b;
    --border: #dfe3ea;
    --radius: 14px;
}

html, body, .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--bg) !important;
}

/* --- Hide chrome --- */
#MainMenu, footer, header { visibility: hidden; height: 0; }
.stDeployButton { display: none; }
.block-container { padding-top: 1.5rem !important; max-width: 800px !important; }

/* ================================================================
   SIDEBAR  –  dark navy, proper inner styling
   ================================================================ */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #007a9a 0%, #005f78 100%) !important;
    padding-top: 0 !important;
}

/* All text inside sidebar → light */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h4,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] .stCaption p {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stMarkdown h4 {
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #ffffff !important;
    font-weight: 700 !important;
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 0.8rem 0;
}

/* Selectbox */
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Toggle */
section[data-testid="stSidebar"] .stToggle label span {
    color: #cbd5e1 !important;
}

/* Button */
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] button {
    background: rgba(208,64,32,0.12) !important;
    border: 1px solid rgba(208,64,32,0.3) !important;
    color: #e05535 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] button:hover {
    background: rgba(208,64,32,0.22) !important;
}

/* Caption */
section[data-testid="stSidebar"] .stCaption p {
    color: rgba(255,255,255,0.3) !important;
    font-size: 0.65rem !important;
}

/* --- Sidebar logo block --- */
.sb-logo-block {
    text-align: center;
    padding: 1.5rem 1rem 0.6rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 0.4rem;
}
.sb-logo-block .sb-name {
    color: #ffffff;
    font-weight: 700;
    font-size: 1rem;
    margin-top: 0.55rem;
    letter-spacing: 0.4px;
}
.sb-logo-block .sb-sub {
    color: rgba(255,255,255,0.4);
    font-size: 0.68rem;
    font-weight: 300;
}



/* ================================================================
   CHAT MESSAGES
   ================================================================ */
.stChatMessage {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    margin-bottom: 0.55rem !important;
}
.stChatMessage p,
.stChatMessage li,
.stChatMessage span:not(.stButton span) {
    color: #1e293b !important;
}
[data-testid="stChatMessage"][aria-label="user"] {
    background: #eef7f9 !important;
    border-color: #b2dce6 !important;
}

/* Chat input bar */
.stChatInput > div {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}
.stChatInput > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(0,136,176,0.10) !important;
}

/* Audio elements */
audio {
    height: 32px !important;
    border-radius: 8px !important;
    width: 100%;
}

/* Source tags */
.src-tag {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.7rem;
    color: var(--text-muted);
    margin: 2px 3px;
}

/* ================================================================
   SPEAKER BUTTON (per-message TTS) — inside chat bubbles
   ================================================================ */
.stChatMessage button {
    background: #f1f5f9 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.25rem 0.65rem !important;
    font-size: 0.78rem !important;
    color: var(--text-muted) !important;
    cursor: pointer;
    transition: all 0.15s;
}
.stChatMessage button:hover {
    background: #e2e8f0 !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
#  SESSION STATE
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "tts_audio" not in st.session_state:
    st.session_state.tts_audio = {}  # msg_index → audio bytes


# =====================================================================
#  SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="sb-logo-block">
        <div class="sb-name">E-Wslni</div>
        <div class="sb-sub">Autonomous Campus Guide</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Voice")
    voice_option = st.selectbox(
        "TTS Voice",
        options=[
            "fr-FR-DeniseNeural",
            "fr-FR-HenriNeural",
            "en-US-AriaNeural",
            "en-US-GuyNeural",
        ],
        format_func=lambda v: {
            "fr-FR-DeniseNeural": "Denise (FR, female)",
            "fr-FR-HenriNeural": "Henri (FR, male)",
            "en-US-AriaNeural": "Aria (EN, female)",
            "en-US-GuyNeural": "Guy (EN, male)",
        }[v],
    )
    auto_tts = st.toggle("Auto-play voice replies", value=False,
                          help="Automatically speak every assistant response")

    st.markdown("---")
    st.markdown("#### Display")
    show_sources = st.toggle("Show sources", value=False)

    st.markdown("---")
    st.markdown("#### Microphone")
    audio_input = st.audio_input(
        "Press to record, press again to stop",
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.greeted = False
        st.session_state.tts_audio = {}
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Gemini, Whisper & Edge TTS")


# =====================================================================
#  BACKEND
# =====================================================================
client, collection = init_backend()


# =====================================================================
#  WELCOME
# =====================================================================
if not st.session_state.greeted:
    welcome = ("Bonjour, je suis E-Wslni, votre assistant virtuel pour l'EMINES et l'UM6P. "
               "Posez-moi vos questions sur les programmes, l'admission, le campus "
               "ou toute autre information.")
    st.session_state.messages.append({"role": "assistant", "content": welcome})
    st.session_state.greeted = True


# =====================================================================
#  PROCESS VOICE INPUT (from sidebar mic)
# =====================================================================
if audio_input is not None:
    audio_bytes = audio_input.getvalue()
    audio_hash = hash(audio_bytes)
    if st.session_state.get("_audio_hash") != audio_hash:
        st.session_state["_audio_hash"] = audio_hash
        with st.spinner("Transcription en cours..."):
            transcription = transcribe_audio(audio_bytes)
        if transcription:
            st.session_state.messages.append({"role": "user", "content": transcription})
            with st.spinner("Recherche..."):
                answer, sources = ask(client, collection, transcription,
                                      st.session_state.conversation_history)
            st.session_state.conversation_history.append((transcription, answer))
            msg = {"role": "assistant", "content": answer, "sources": list(sources)}
            if auto_tts:
                msg["audio"] = tts_bytes(answer, voice_option)
            st.session_state.messages.append(msg)
            st.rerun()


# =====================================================================
#  DISPLAY CHAT HISTORY + per-message speaker buttons
# =====================================================================
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show audio player if already generated
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")

        # Show TTS generated on-demand
        if i in st.session_state.tts_audio:
            st.audio(st.session_state.tts_audio[i], format="audio/mp3")

        # Speaker button for every assistant message (hybrid mode)
        if msg["role"] == "assistant" and not msg.get("audio") and i not in st.session_state.tts_audio:
            if st.button("Listen", key=f"tts_{i}"):
                with st.spinner(""):
                    audio_data = tts_bytes(msg["content"], voice_option)
                st.session_state.tts_audio[i] = audio_data
                st.rerun()

        # Sources
        if msg.get("sources") and show_sources:
            st.markdown(
                " ".join(f'<span class="src-tag">{s}</span>' for s in msg["sources"]),
                unsafe_allow_html=True,
            )


# =====================================================================
#  TEXT INPUT
# =====================================================================
user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(""):
            answer, sources = ask(client, collection, user_input,
                                  st.session_state.conversation_history)
            st.session_state.conversation_history.append((user_input, answer))

        st.markdown(answer)

        audio_data = None
        if auto_tts:
            with st.spinner(""):
                audio_data = tts_bytes(answer, voice_option)
            st.audio(audio_data, format="audio/mp3")

        if show_sources:
            st.markdown(
                " ".join(f'<span class="src-tag">{s}</span>' for s in sources),
                unsafe_allow_html=True,
            )

    msg = {"role": "assistant", "content": answer, "sources": list(sources)}
    if audio_data:
        msg["audio"] = audio_data
    st.session_state.messages.append(msg)
