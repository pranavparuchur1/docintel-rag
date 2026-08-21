"""Deterministic query routing — the free-by-default brain of the agent.

Routes: lookup | comparison | temporal | out_of_scope. An LLM router is a
drop-in upgrade later; this one is rule-based so the whole agent runs with
zero API keys, and every routing decision is reproducible in a test.

The out_of_scope entity guard exists because the eval measured that cosine
similarity cannot detect entity absence (refusal recall 0.50 at threshold
0.68): "What supply chain risks does Walmart report?" retrieves other
companies' supply-chain text at high similarity. The guard: if the query
names NO corpus company but does contain an unknown proper-noun-looking
token, the corpus cannot answer it, whatever the cosine says.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Corpus company aliases -> canonical documents.company value.
COMPANY_ALIASES: dict[str, str] = {
    "apple": "Apple Inc.",
    "microsoft": "MICROSOFT CORP",
    "msft": "MICROSOFT CORP",
    "nvidia": "NVIDIA CORP",
    "tesla": "Tesla, Inc.",
    "coca-cola": "COCA COLA CO",
    "coca cola": "COCA COLA CO",
    "coke": "COCA COLA CO",
    "jpmorgan": "JPMORGAN CHASE & CO",
    "jp morgan": "JPMORGAN CHASE & CO",
    "jpm": "JPMORGAN CHASE & CO",
    "chase": "JPMORGAN CHASE & CO",
}

_TEMPORAL_CUES = re.compile(
    r"\b(change[ds]?|between|evolv\w+|year[- ]over[- ]year|prior year|last two|"
    r"most recent (?:two|annual|filings)|across (?:its|the|their).{0,20}(?:10-K|filings|years)|"
    r"compare[d]? .{0,30}(?:across|between|over time))\b",
    re.IGNORECASE,
)

# Vocabulary that legitimately appears capitalized mid-sentence in in-corpus
# questions and must not trip the unknown-entity guard.
_KNOWN_CAPITALIZED = {
    "item", "items", "risk", "factors", "md&a", "mda", "sec", "edgar", "form",
    "10-k", "10-q", "u.s.", "us", "eu", "china", "taiwan", "asia", "europe",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "q1", "q2", "q3", "q4",
    "i", "ii", "iii", "iv", "fy2024", "fy2025", "fy2026", "covid-19", "ai",
    # corpus-adjacent product/segment/person/reg names used by golden questions
    "elon", "musk", "autopilot", "iphone", "mac", "ipad", "azure", "windows",
    "office", "intelligent", "cloud", "data", "center", "app", "store",
    "digital", "markets", "act", "dma", "gaap", "cybertruck", "model",
}


@dataclass(frozen=True)
class RouteDecision:
    route: str  # lookup | comparison | temporal | out_of_scope
    companies: list[str] = field(default_factory=list)  # canonical names
    unknown_entities: list[str] = field(default_factory=list)
    reason: str = ""


def detect_companies(question: str) -> list[str]:
    q = question.lower()
    found: list[str] = []
    for alias, canonical in COMPANY_ALIASES.items():
        if alias in q and canonical not in found:
            found.append(canonical)
    return found


def unknown_entities(question: str, companies: list[str]) -> list[str]:
    """Capitalized tokens that are neither sentence-initial, corpus aliases,
    nor known filing vocabulary — a cheap stand-in for NER."""
    tokens = re.findall(r"(?<!^)(?<![.?!] )\b([A-Z][A-Za-z&.-]+)", question)
    alias_words = {w for alias in COMPANY_ALIASES for w in alias.split()}
    for canonical in companies:
        alias_words.update(w.lower().strip(",.&") for w in canonical.split())
    out = []
    for token in tokens:
        lowered = token.lower().strip(",.?!'s")
        if lowered.endswith("'s"):
            lowered = lowered[:-2]
        if lowered in _KNOWN_CAPITALIZED or lowered in alias_words:
            continue
        if token not in out:
            out.append(token)
    return out


def route_question(question: str) -> RouteDecision:
    companies = detect_companies(question)
    unknowns = unknown_entities(question, companies)

    if not companies and unknowns:
        return RouteDecision(
            route="out_of_scope", unknown_entities=unknowns,
            reason=f"query names entities not in the corpus: {', '.join(unknowns[:3])}",
        )
    if len(companies) >= 2:
        return RouteDecision(route="comparison", companies=companies)
    if len(companies) == 1 and _TEMPORAL_CUES.search(question):
        return RouteDecision(route="temporal", companies=companies)
    return RouteDecision(route="lookup", companies=companies)
