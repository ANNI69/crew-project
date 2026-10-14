import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, reply="The answer is 42."):
        self.reply = reply
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return FakeResponse(self.reply)


class FakeChat:
    def __init__(self, reply):
        self.completions = FakeCompletions(reply)


class FakeLLM:
    def __init__(self, reply="The answer is 42."):
        self.chat = FakeChat(reply)


class ExplodingLLM(FakeLLM):
    def __init__(self):
        super().__init__(RuntimeError("Groq down"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "fake")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.2")
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def fake_llm(monkeypatch):
    llm = FakeLLM("Mercury is the closest planet to the Sun.")
    monkeypatch.setattr("app.rag.get_llm", lambda settings: llm)
    return llm
