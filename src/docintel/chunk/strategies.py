"""The three chunk strategies, all behind ChunkStrategy.

Budgets are in cl100k tokens, kept well under the embedding model's 512-token
window (see tokens.py for why the counter is an approximation).
"""

from __future__ import annotations

from docintel.chunk.base import Chunk, ChunkStrategy
from docintel.chunk.tokens import count_tokens, encoding
from docintel.parse import ParsedDoc

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


class FixedSizeChunker(ChunkStrategy):
    """Sliding window over tokens: brutal but a necessary baseline. Overlap
    exists so a fact straddling a boundary is whole in at least one chunk."""

    name = "fixed"

    def __init__(self, size_tokens: int = 350, overlap_tokens: int = 60, min_tokens: int = 24):
        if overlap_tokens >= size_tokens:
            raise ValueError("overlap must be smaller than chunk size")
        self.size = size_tokens
        self.overlap = overlap_tokens
        self.min_tokens = min_tokens

    @property
    def params(self) -> dict:
        return {
            "size_tokens": self.size,
            "overlap_tokens": self.overlap,
            "min_tokens": self.min_tokens,
        }

    def split(self, doc: ParsedDoc) -> list[Chunk]:
        enc = encoding()
        ids = enc.encode(doc.text)
        step = self.size - self.overlap
        spans: list[tuple[int, str]] = []
        char_offset = 0
        for i in range(0, len(ids), step):
            window_text = enc.decode(ids[i : i + self.size])
            spans.append((char_offset, window_text))
            # advance offset by the stride's decoded length (approximate at
            # UTF-8 boundaries; used only for section attribution)
            char_offset += len(enc.decode(ids[i : i + step]))
        return self._finalize(doc, spans, self.min_tokens)


def _split_span(
    text: str, start: int, end: int, max_tokens: int, separators: tuple[str, ...]
) -> list[tuple[int, int]]:
    """Recursively split [start, end) of `text` at the coarsest separator that
    keeps pieces under budget, then greedily re-merge neighbors up to budget."""
    span_text = text[start:end]
    if count_tokens(span_text) <= max_tokens:
        return [(start, end)]

    sep = next((s for s in separators if s in span_text), None)
    if sep is None:
        # No separator left: hard-split by tokens.
        enc = encoding()
        ids = enc.encode(span_text)
        pieces: list[tuple[int, int]] = []
        offset = start
        for i in range(0, len(ids), max_tokens):
            piece = enc.decode(ids[i : i + max_tokens])
            pieces.append((offset, min(end, offset + len(piece))))
            offset += len(piece)
        return pieces

    remaining = separators[separators.index(sep) :]
    finer = remaining[1:]

    cuts: list[tuple[int, int]] = []
    prev = start
    idx = span_text.find(sep)
    while idx != -1:
        cut = start + idx + len(sep)
        cuts.append((prev, cut))
        prev = cut
        idx = span_text.find(sep, cut - start)
    if prev < end:
        cuts.append((prev, end))

    expanded: list[tuple[int, int]] = []
    for s, e in cuts:
        if count_tokens(text[s:e]) > max_tokens:
            expanded.extend(_split_span(text, s, e, max_tokens, finer))
        else:
            expanded.append((s, e))

    # Greedy merge: pack adjacent pieces back together up to the budget so we
    # emit few, dense chunks instead of one chunk per paragraph.
    merged: list[tuple[int, int]] = []
    current_start, current_tokens = None, 0
    for s, e in expanded:
        piece_tokens = count_tokens(text[s:e])
        if current_start is None:
            current_start, current_end, current_tokens = s, e, piece_tokens
        elif current_tokens + piece_tokens <= max_tokens:
            current_end, current_tokens = e, current_tokens + piece_tokens
        else:
            merged.append((current_start, current_end))
            current_start, current_end, current_tokens = s, e, piece_tokens
    if current_start is not None:
        merged.append((current_start, current_end))
    return merged


class RecursiveChunker(ChunkStrategy):
    """Split at natural boundaries (paragraph > line > sentence > word),
    merging back up to the budget. No overlap: boundaries fall where the text
    itself breaks, which is where straddling facts are least likely."""

    name = "recursive"

    def __init__(self, max_tokens: int = 350, min_tokens: int = 24):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    @property
    def params(self) -> dict:
        return {"max_tokens": self.max_tokens, "min_tokens": self.min_tokens}

    def split(self, doc: ParsedDoc) -> list[Chunk]:
        spans = _split_span(doc.text, 0, len(doc.text), self.max_tokens, DEFAULT_SEPARATORS)
        return self._finalize(doc, [(s, doc.text[s:e]) for s, e in spans], self.min_tokens)


class SectionAwareChunker(ChunkStrategy):
    """Recursive splitting, but never across a detected section boundary: a
    chunk about Risk Factors cannot bleed into MD&A, and every chunk carries
    its section label for citation."""

    name = "section_aware"

    def __init__(self, max_tokens: int = 350, min_tokens: int = 24):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    @property
    def params(self) -> dict:
        return {"max_tokens": self.max_tokens, "min_tokens": self.min_tokens}

    def split(self, doc: ParsedDoc) -> list[Chunk]:
        boundaries = sorted({0, len(doc.text), *(s.start for s in doc.sections)})
        spans: list[tuple[int, str]] = []
        for region_start, region_end in zip(boundaries, boundaries[1:], strict=False):
            for s, e in _split_span(
                doc.text, region_start, region_end, self.max_tokens, DEFAULT_SEPARATORS
            ):
                spans.append((s, doc.text[s:e]))
        return self._finalize(doc, spans, self.min_tokens)


def get_strategies(
    names: list[str] | None = None, max_tokens: int | None = None
) -> list[ChunkStrategy]:
    """max_tokens overrides the token budget (size_tokens for fixed) — used to
    demonstrate that a param change creates a new params_hash and therefore a
    new index version instead of polluting an existing one."""
    if max_tokens is None:
        instances = (FixedSizeChunker(), RecursiveChunker(), SectionAwareChunker())
    else:
        instances = (
            FixedSizeChunker(size_tokens=max_tokens),
            RecursiveChunker(max_tokens=max_tokens),
            SectionAwareChunker(max_tokens=max_tokens),
        )
    registry: dict[str, ChunkStrategy] = {s.name: s for s in instances}
    if names is None or names == ["all"]:
        return list(registry.values())
    missing = [n for n in names if n not in registry]
    if missing:
        raise ValueError(f"unknown strategies {missing}; available: {sorted(registry)}")
    return [registry[n] for n in names]
