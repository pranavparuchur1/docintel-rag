"""Embedding providers behind one interface.

Default is local + free: sentence-transformers running bge-small on CPU.
OpenAI is opt-in via EMBEDDING_PROVIDER=openai and never required. The model
import/load is lazy so that constructing a provider (config validation, CLI
--help, unit tests) costs nothing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from docintel.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    model_name: str

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class LocalBgeProvider(EmbeddingProvider):
    # bge v1.5 models are trained with an instruction prefix on the QUERY side
    # only; passages are embedded bare. Getting this wrong silently costs recall.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            logger.info("loading %s (first use; downloads to HF cache once)", self.model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    def prepare_query(self, text: str) -> str:
        return self.QUERY_PREFIX + text

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(
            texts, batch_size=len(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_passages([self.prepare_query(text)])[0]


class OpenAIProvider(EmbeddingProvider):
    """Opt-in paid provider (EMBEDDING_PROVIDER=openai)."""

    URL = "https://api.openai.com/v1/embeddings"
    DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small") -> None:
        if model_name not in self.DIMS:
            raise ValueError(f"unknown OpenAI embedding model {model_name!r}")
        self.model_name = model_name
        self._api_key = api_key

    @property
    def dim(self) -> int:
        return self.DIMS[self.model_name]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        import httpx

        response = httpx.post(
            self.URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self.model_name, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_passages([text])[0]


def get_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIProvider(settings.openai_api_key, "text-embedding-3-small")
    return LocalBgeProvider(settings.embedding_model)
