import io
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .chunker import chunk_text
from .config import get_settings
from .embeddings import get_embedder
from .rag import answer_question
from .vectorstore import VectorStore

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.store = VectorStore(Path(settings.data_dir) / "store")
    app.state.embedder = get_embedder(settings)
    yield


app = FastAPI(title="Minimal RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    name: str = "document"
    text: str


class AskInput(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None


def _extract_text(name: str, raw: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return raw.decode("utf-8", errors="replace")


def _ingest(name: str, text: str) -> dict:
    settings = app.state.settings
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail=f"Document '{name}' contains no text.")
    doc_id = uuid.uuid4().hex
    embeddings = app.state.embedder.embed(chunks)
    app.state.store.add(doc_id, name, chunks, embeddings)
    return {"doc_id": doc_id, "name": name, "chunks": len(chunks)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents")
def ingest_file(file: UploadFile) -> dict:
    name = file.filename or "document"
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type for '{name}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    return _ingest(name, _extract_text(name, file.file.read()))


@app.post("/documents/text")
def ingest_text(body: TextInput) -> dict:
    return _ingest(body.name, body.text)


@app.get("/documents")
def list_documents() -> list[dict]:
    return app.state.store.documents()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not app.state.store.delete(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": doc_id}


@app.post("/ask")
def ask(body: AskInput) -> dict:
    settings = app.state.settings
    query_embedding = app.state.embedder.embed([body.question])[0]
    top_k = body.top_k or settings.top_k
    hits = app.state.store.search(query_embedding, top_k=top_k)
    hits = [h for h in hits if h["score"] >= settings.similarity_threshold]
    try:
        return answer_question(body.question, hits, settings)
    except Exception as exc:  # Groq/network failure
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
