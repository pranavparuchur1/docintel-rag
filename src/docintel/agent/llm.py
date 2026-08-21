"""Answer generation behind one interface.

Default is LLM_PROVIDER=none: extractive retrieval-only mode — the strongest
evidence, quoted with citations, clearly labeled. Zero keys, zero cost, fully
reproducible. Anthropic (official SDK) and OpenAI are opt-in via env.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from docintel.config import Settings
from docintel.retrieve.service import RetrievedChunk

SYSTEM_PROMPT = (
    "You answer questions about SEC filings using ONLY the numbered context "
    "chunks provided. Cite every claim inline as [n] using the chunk numbers. "
    "If the context does not contain the answer, say exactly: 'The provided "
    "filings do not cover this.' Never use outside knowledge."
)


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    tokens_in: int
    tokens_out: int


def format_citation(chunk: RetrievedChunk) -> str:
    section = chunk.section or "no section"
    return (
        f"{chunk.company} {chunk.form_type} {chunk.document_id} — "
        f"{section} — chunk {chunk.chunk_id}"
    )


def _context_blocks(chunks: list[RetrievedChunk], max_chunks: int = 6) -> str:
    blocks = []
    for i, chunk in enumerate(chunks[:max_chunks], start=1):
        blocks.append(f"[{i}] ({format_citation(chunk)})\n{chunk.text}")
    return "\n\n".join(blocks)


class Answerer(ABC):
    provider = "none"

    @abstractmethod
    def answer(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer: ...


class ExtractiveAnswerer(Answerer):
    """Retrieval-only mode: no generation, no hallucination surface at all."""

    provider = "none"

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        lines = [
            "[retrieval-only mode — no LLM configured; strongest evidence, verbatim]",
        ]
        for i, chunk in enumerate(chunks[:4], start=1):
            excerpt = " ".join(chunk.text.split())
            if len(excerpt) > 500:
                excerpt = excerpt[:500] + "…"
            lines.append(f"[{i}] {format_citation(chunk)}\n    \"{excerpt}\"")
        return GeneratedAnswer(text="\n\n".join(lines), tokens_in=0, tokens_out=0)


class AnthropicAnswerer(Answerer):
    provider = "anthropic"

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _load(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # resolves key from env/profile
        return self._client

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        response = self._load().messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n\n{_context_blocks(chunks)}\n\n"
                    f"Question: {question}",
                }
            ],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return GeneratedAnswer(
            text=text,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
        )


class OpenAIAnswerer(Answerer):
    provider = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._api_key = api_key

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        import httpx

        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n\n{_context_blocks(chunks)}\n\n"
                        f"Question: {question}",
                    },
                ],
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return GeneratedAnswer(
            text=data["choices"][0]["message"]["content"],
            tokens_in=data["usage"]["prompt_tokens"],
            tokens_out=data["usage"]["completion_tokens"],
        )


def get_answerer(settings: Settings) -> Answerer:
    if settings.llm_provider == "anthropic":
        return AnthropicAnswerer(settings.llm_model or "claude-opus-5")
    if settings.llm_provider == "openai":
        return OpenAIAnswerer(settings.openai_api_key, settings.llm_model or "gpt-4o-mini")
    return ExtractiveAnswerer()
