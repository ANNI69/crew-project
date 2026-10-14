# Document Q&A — Minimal RAG over FastAPI + Groq + React

Ingest PDF/TXT/Markdown documents, retrieve relevant chunks with local
embeddings, and answer questions with a Groq-hosted LLM. Answers include
source document names and excerpts; when the documents don't contain the
answer, the API says so explicitly.

## Architecture

```
frontend/            React (Vite) UI: upload docs, ask questions, view sources
backend/
  app/
    main.py          FastAPI app: /documents, /ask endpoints
    chunker.py       paragraph-aware text splitting
    embeddings.py    fastembed (BAAI/bge-small-en-v1.5, local ONNX, CPU)
    vectorstore.py   persistent numpy cosine store
    rag.py           retrieval + Groq (Llama 3.3) grounded generation
    config.py        settings from environment variables
  tests/             pytest suite: chunking, retrieval, API, failure cases
eval/
  questions.json     12 evaluation questions (answerable/unanswerable/multi)
  run_eval.py        scoring harness against a running backend
sample_docs/         sample corpus used by the evaluation
docker-compose.yml   backend (uvicorn) + frontend (nginx)
```

- **Embeddings** run locally via `fastembed` (no API key, model downloaded on
  first run and cached in the Docker `model-cache` volume).
- **Generation** uses the Groq API (`llama-3.3-70b-versatile` by default).
- **Vector store** is a small numpy cosine index persisted under `DATA_DIR`.

## Setup (local)

```bash
# 1. Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp ../.env.example ../.env        # then put your real GROQ_API_KEY in .env
uvicorn app.main:app --reload --port 8000

# 2. Frontend (new terminal)
cd frontend
npm install
npm run dev                       # http://localhost:5173, proxies /api -> :8000
```

Get a Groq API key at <https://console.groq.com/keys>.

## Setup (Docker)

```bash
cp .env.example .env              # set GROQ_API_KEY
docker compose up --build
# frontend: http://localhost:5173   backend API: http://localhost:8000/docs
```

The compose file persists ingested documents (`rag-data` volume) and the
embedding model cache (`model-cache` volume).

## API

| Method | Path                        | Description                                    |
| ------ | --------------------------- | ---------------------------------------------- |
| POST   | `/documents`                | multipart file upload (`.pdf`, `.txt`, `.md`)  |
| POST   | `/documents/text`           | ingest raw text: `{"name": "...", "text": "..."}` |
| GET    | `/documents`                | list ingested documents                        |
| DELETE | `/documents/{doc_id}`       | remove a document and its chunks               |
| POST   | `/ask`                      | `{"question": "..."}` → answer + sources       |
| GET    | `/health`                   | liveness check                                 |

`POST /ask` response:

```json
{
  "answer": "New hires receive a laptop voucher of 2,400 USD.",
  "grounded": true,
  "sources": [
    {"document": "handbook.md", "excerpt": "All new hires receive a laptop voucher of 2,400 USD...", "score": 0.61}
  ]
}
```

If no retrieved chunk passes `SIMILARITY_THRESHOLD` (default `0.35`), the LLM
is **not** called and the API returns `grounded: false` with:

> "The requested information is unavailable in the provided documents."

## Configuration

All via environment variables (see `.env.example`): `GROQ_API_KEY`,
`GROQ_MODEL`, `EMBEDDINGS_PROVIDER`, `EMBEDDINGS_MODEL`, `CHUNK_SIZE`,
`CHUNK_OVERLAP`, `TOP_K`, `SIMILARITY_THRESHOLD`, `DATA_DIR`.

## Tests

```bash
cd backend
source ../.venv/bin/activate       # or your venv
pytest -q
```

26 automated tests cover chunking, retrieval relevance and thresholds, store
persistence, API behavior (TXT/Markdown/PDF ingestion, listing, deletion,
sources), and failure cases (unsupported file type, empty document, empty
question, LLM outage → 502, questions with no evidence → no LLM call). Tests
run offline with a deterministic fake embedder and a stubbed LLM — no API key
or model download needed.

## Evaluation

With the backend running and `GROQ_API_KEY` set:

```bash
python eval/run_eval.py --base-url http://localhost:8000
```

This ingests the three `sample_docs/` files and scores 12 questions:

- **5 answerable** — must be grounded and contain the expected fact
- **4 unanswerable** — must report the information as unavailable
- **3 multi-evidence** — must combine facts and cite ≥ 2 source chunks

The script prints a PASS/FAIL table and exits non-zero on any failure.
