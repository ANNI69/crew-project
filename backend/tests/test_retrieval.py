import numpy as np
import pytest

from app.config import Settings
from app.embeddings import FakeEmbedder
from app.rag import answer_question
from app.vectorstore import VectorStore

DOCS = {
    "planets.txt": "Mercury is the closest planet to the Sun. Venus is the second planet.",
    "cooking.md": "To make pasta, boil water with salt, then cook the noodles for eight minutes.",
}


@pytest.fixture()
def store(tmp_path):
    embedder = FakeEmbedder()
    store = VectorStore(tmp_path / "store")
    for name, text in DOCS.items():
        chunks = [text]
        store.add(doc_id=name, doc_name=name, chunks=chunks, embeddings=embedder.embed(chunks))
    return store


def test_retrieval_finds_relevant_document(store):
    embedder = FakeEmbedder()
    query = embedder.embed(["Which planet is closest to the Sun?"])[0]
    hits = store.search(query, top_k=2)
    assert hits[0]["doc_name"] == "planets.txt"
    assert hits[0]["score"] > hits[1]["score"]


def test_unrelated_query_scores_low(store):
    embedder = FakeEmbedder()
    query = embedder.embed(["zzz qqq xxx unrelated gibberish tokens"])[0]
    hits = store.search(query, top_k=2)
    assert hits[0]["score"] < 0.2  # below the configured answer threshold


def test_empty_store_returns_nothing(tmp_path):
    store = VectorStore(tmp_path / "empty")
    embedder = FakeEmbedder()
    assert store.search(embedder.embed(["anything"])[0]) == []


def test_persistence_roundtrip(tmp_path):
    embedder = FakeEmbedder()
    path = tmp_path / "persisted"
    store = VectorStore(path)
    store.add("d1", "d1.txt", ["hello world"], embedder.embed(["hello world"]))

    reloaded = VectorStore(path)
    assert len(reloaded.records) == 1
    hits = reloaded.search(embedder.embed(["hello world"])[0], top_k=1)
    assert hits[0]["text"] == "hello world"


def test_delete_document(store):
    doc_id = store.documents()[0]["doc_id"]
    assert store.delete(doc_id) is True
    assert store.delete(doc_id) is False
    assert len(store.documents()) == 1


def test_unanswerable_question_skips_llm(store):
    embedder = FakeEmbedder()
    settings = Settings(similarity_threshold=0.99)  # nothing can pass
    query = embedder.embed(["What is the airspeed velocity of an unladen swallow?"])[0]
    hits = store.search(query, top_k=2)
    hits = [h for h in hits if h["score"] >= settings.similarity_threshold]

    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise AssertionError("LLM must not be called without evidence")

    result = answer_question("unanswerable", hits, settings, llm=Boom)
    assert result["grounded"] is False
    assert result["sources"] == []
    assert "unavailable" in result["answer"].lower()


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Stub LLM that records nothing and returns a canned completion."""

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                class Response:
                    choices = [type("C", (), {"message": _FakeMessage("Pasta answer.")})()]

                return Response()


def test_grounded_answer_includes_sources(store):
    embedder = FakeEmbedder()
    settings = Settings(similarity_threshold=0.2)
    query = embedder.embed(["How do I cook pasta?"])[0]
    hits = [h for h in store.search(query, top_k=2) if h["score"] >= settings.similarity_threshold]
    result = answer_question("How do I cook pasta?", hits, settings, llm=_FakeLLM())
    assert result["grounded"] is True
    assert result["sources"][0]["document"] == "cooking.md"
    assert "boil water" in result["sources"][0]["excerpt"]
    assert isinstance(result["sources"][0]["score"], float)
