# E-Wslni — EMINES / UM6P Campus Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about **EMINES – School of Industrial Management** and the **Université Mohammed VI Polytechnique (UM6P)**. Designed to be integrated into an autonomous campus-guide robot.

## Features

- **Semantic RAG pipeline** — Retrieves relevant knowledge chunks from a ChromaDB vector store and generates grounded answers with Google Gemini.
- **Multilingual** — Responds in the same language as the question (French, English, Arabic…).
- **Voice I/O** — Speech-to-text with OpenAI Whisper, text-to-speech with Microsoft Edge TTS (free, no API key).
- **Web interface** — Streamlit app with sidebar voice controls, per-message "Listen" button, auto-TTS toggle, and source display.
- **Terminal mode** — Text or voice chat loops for headless / robot deployment.
- **Conversation memory** — Keeps the last 5 exchanges for context-aware follow-up answers.

## Architecture

```
data/  (Markdown + JSON sources)
  │
  ▼
chunker.py   →  Semantic header-aware chunking
  │
  ▼
embedder.py  →  Gemini embeddings  →  ChromaDB (chroma_db/)
  │
  ▼
chatbot.py   →  Terminal chatbot (text / voice)
app.py       →  Streamlit web UI
```

## Project Structure

```
├── chunker.py          # Step 1 — Load and semantically chunk Markdown/JSON docs
├── embedder.py         # Step 2 — Embed chunks with Gemini and store in ChromaDB
├── chatbot.py          # Step 3 — Terminal chatbot (text + voice modes)
├── app.py              # Streamlit web interface
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed)
├── data/               # Source documents (Markdown, JSON)
└── chroma_db/          # ChromaDB vector store (auto-generated)
```

## Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/apikey) (free tier is sufficient)
- A working microphone and speakers (for voice mode)

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/e-wslni.git
cd e-wslni

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create the .env file with your API key
echo GOOGLE_API_KEY=your_key_here > .env
```

## Usage

### Build the vector store (first time only)

```bash
# Chunk the source documents
python chunker.py

# Embed and store in ChromaDB
python embedder.py
```

### Terminal chatbot

```bash
python chatbot.py
# Choose 1 for text mode, 2 for voice mode
```

### Web interface

```bash
streamlit run app.py
```

Open the URL printed in the terminal (default: `http://localhost:8501`).

## Configuration

| Variable | Location | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | `.env` | — | Gemini API key |
| `LLM_MODEL` | `chatbot.py` / `app.py` | `gemini-2.5-flash` | Generation model |
| `EMBEDDING_MODEL` | `embedder.py` | `gemini-embedding-001` | Embedding model |
| `TOP_K` | `chatbot.py` / `app.py` | `5` | Number of chunks retrieved per query |
| `TTS_VOICE` | `app.py` | `fr-FR-DeniseNeural` | Default Edge TTS voice |

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 001 |
| Vector Store | ChromaDB |
| STT | OpenAI Whisper (base) |
| TTS | Microsoft Edge TTS |
| Web UI | Streamlit |

## License

This project is for educational and research purposes at UM6P / EMINES.
