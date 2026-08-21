"""Token counting for chunk sizing and stats.

Uses tiktoken's cl100k_base as a fast, deterministic counter. It approximates
(but is not identical to) the WordPiece tokenizer of the embedding model, so
chunk budgets are set conservatively below the model's 512-token window.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoding().encode(text))
