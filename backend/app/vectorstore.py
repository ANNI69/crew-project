import json
import uuid
from pathlib import Path

import numpy as np


class VectorStore:
    """Minimal persistent vector store (numpy cosine over normalized embeddings)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: list[dict] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self._load()

    def add(self, doc_id: str, doc_name: str, chunks: list[str], embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        for index, (chunk, _) in enumerate(zip(chunks, embeddings)):
            self.records.append(
                {
                    "id": uuid.uuid4().hex,
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "text": chunk,
                    "index": index,
                }
            )
        self.matrix = (
            np.vstack([self.matrix, embeddings]) if self.matrix.size else embeddings
        )
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[dict]:
        if not self.records:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        scores = self.matrix @ query
        order = np.argsort(scores)[::-1][:top_k]
        return [{"score": float(scores[i]), **self.records[i]} for i in order]

    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for record in self.records:
            doc = docs.setdefault(
                record["doc_id"],
                {"doc_id": record["doc_id"], "name": record["doc_name"], "chunks": 0},
            )
            doc["chunks"] += 1
        return list(docs.values())

    def delete(self, doc_id: str) -> bool:
        keep = [i for i, r in enumerate(self.records) if r["doc_id"] != doc_id]
        if len(keep) == len(self.records):
            return False
        self.records = [self.records[i] for i in keep]
        self.matrix = self.matrix[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        self.save()
        return True

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "records.json").write_text(json.dumps(self.records))
        np.save(self.path / "embeddings.npy", self.matrix)

    def _load(self) -> None:
        records_file = self.path / "records.json"
        embeddings_file = self.path / "embeddings.npy"
        if records_file.exists() and embeddings_file.exists():
            self.records = json.loads(records_file.read_text())
            self.matrix = np.load(embeddings_file)
