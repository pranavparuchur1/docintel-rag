"""Deterministic embedding provider for tests: 8-dim vectors derived from the
text hash. No network, no model download, stable across runs."""

import hashlib

from docintel.embed.providers import EmbeddingProvider


class FakeProvider(EmbeddingProvider):
    model_name = "fake-embedder-8d"

    def __init__(self):
        self.calls = 0

    @property
    def dim(self) -> int:
        return 8

    def embed_passages(self, texts):
        self.calls += len(texts)
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vec = [b / 255.0 for b in digest[:8]]
            norm = sum(x * x for x in vec) ** 0.5
            out.append([x / norm for x in vec])
        return out

    def embed_query(self, text):
        return self.embed_passages([text])[0]
