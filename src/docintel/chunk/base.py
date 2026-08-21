"""ChunkStrategy interface. Every strategy declares its params; the params hash
is part of chunk identity (and later of index-version identity), so changing a
knob can never silently mix differently-chunked text in one index."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from bisect import bisect_right
from dataclasses import dataclass

from docintel.chunk.tokens import count_tokens
from docintel.parse import ParsedDoc


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str
    section: str | None  # human-readable label, e.g. "Item 1A. Risk Factors"
    token_count: int
    content_hash: str


def content_hash(text: str) -> str:
    """SHA-256 of whitespace-normalized text. Normalization means incidental
    reflowing (parser tweaks that only change spacing) does not invalidate
    embeddings; any visible character change does."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class ChunkStrategy(ABC):
    name: str

    @property
    @abstractmethod
    def params(self) -> dict: ...

    @property
    def params_hash(self) -> str:
        canonical = json.dumps(self.params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @abstractmethod
    def split(self, doc: ParsedDoc) -> list[Chunk]: ...

    # --- shared helpers -------------------------------------------------------
    @staticmethod
    def _section_label_at(doc: ParsedDoc, char_offset: int) -> str | None:
        starts = [s.start for s in doc.sections]
        idx = bisect_right(starts, char_offset) - 1
        if idx < 0:
            return None
        section = doc.sections[idx]
        return section.label if char_offset < section.end else None

    @staticmethod
    def _finalize(doc: ParsedDoc, spans: list[tuple[int, str]], min_tokens: int) -> list[Chunk]:
        """Turn (start_offset, text) spans into Chunks, dropping fragments too
        small to be retrievable evidence (bare headings, stray table cells)."""
        chunks: list[Chunk] = []
        for start, text in spans:
            text = text.strip()
            if not text:
                continue
            tokens = count_tokens(text)
            if tokens < min_tokens:
                continue
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    text=text,
                    section=ChunkStrategy._section_label_at(doc, start),
                    token_count=tokens,
                    content_hash=content_hash(text),
                )
            )
        return chunks
