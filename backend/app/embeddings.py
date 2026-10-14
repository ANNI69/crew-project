import hashlib
import re

import numpy as np


class FakeEmbedder:
    """Deterministic hash-based embedder used in tests (no downloads, no network)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
                vec[digest % self.dim] += 1.0
            norm = float(np.linalg.norm(vec))
            vectors.append(vec / norm if norm else vec)
        return np.stack(vectors)


class FastEmbedEmbedder:
    """Local embedding model via fastembed (ONNX, CPU, no training)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = np.array(list(self.model.embed(list(texts))), dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)


def get_embedder(settings):
    if settings.embeddings_provider == "fake":
        return FakeEmbedder()
    return FastEmbedEmbedder(settings.embeddings_model)
