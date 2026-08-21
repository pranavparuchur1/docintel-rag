"""Retrieval planning shared by the LangGraph agent and the eval harness.

Kept separate from the graph so `docintel eval --mode agent` measures exactly
the retrieval the agent performs — same routing, same fan-out, same refusal
bar — without dragging grading/answering into retrieval metrics.
"""

from __future__ import annotations

import re

import psycopg

from docintel.agent.routing import RouteDecision, route_question
from docintel.config import Settings
from docintel.retrieve.fanout import comparison_search, temporal_search
from docintel.retrieve.service import RetrievalResult, Retriever


def refusal_bar(settings: Settings, decision: RouteDecision) -> float:
    """Queries naming a corpus company have an entity anchor; queries without
    one must clear a higher similarity bar, because a generic query can score
    ~0.7 against *something* in 15k chunks of financial prose."""
    if decision.companies:
        return settings.refusal_threshold
    return settings.refusal_threshold_no_entity


def form_hint(question: str) -> str | None:
    """Temporal questions usually name the filing cadence; honoring it stops
    10-Q legs from eating ranking slots when the user asked about 10-Ks (the
    eval measured exactly this failure: temporal recall stuck at 0.25 while
    quarterly chunks displaced annual-report evidence)."""
    if re.search(r"\bannual report|10-Ks?\b", question, re.IGNORECASE):
        return "10-K"
    if re.search(r"\bquarter|10-Qs?\b", question, re.IGNORECASE):
        return "10-Q"
    return None


def plan_retrieve(
    conn: psycopg.Connection,
    retriever: Retriever,
    question: str,
    k: int = 10,
    mode: str = "hybrid",
) -> tuple[RouteDecision, RetrievalResult | None]:
    """Route, then run the route's retrieval strategy. Returns (decision,
    result); result is None only for out_of_scope routes."""
    decision = route_question(question)
    if decision.route == "out_of_scope":
        return decision, None
    if decision.route == "comparison":
        result = comparison_search(retriever, question, decision.companies, k=k, mode=mode)
    elif decision.route == "temporal":
        result = temporal_search(
            conn, retriever, question, decision.companies[0], k=k, mode=mode,
            form_type=form_hint(question),
        )
    else:  # lookup
        companies = decision.companies or None
        result = retriever.search(question, k=k, mode=mode, companies=companies)
    return decision, result
