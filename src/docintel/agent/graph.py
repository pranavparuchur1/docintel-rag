"""The LangGraph agent: route -> retrieve -> grade -> answer | refuse.

Bounded by construction: `attempts` caps retrieval at 2 (one deterministic
rewrite), so the graph cannot loop. Every run — answered or refused — is
logged to query_runs with retrieved chunk ids, scores, route, tokens, and
latency. Export the diagram with `docintel export-graph`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TypedDict

import psycopg
from langgraph.graph import END, START, StateGraph

from docintel.agent.llm import Answerer, format_citation
from docintel.agent.plan import plan_retrieve, refusal_bar
from docintel.agent.routing import RouteDecision
from docintel.config import Settings
from docintel.retrieve.service import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # initial retrieval + one rewritten retry


class AgentState(TypedDict, total=False):
    question: str
    working_question: str  # possibly rewritten
    k: int
    mode: str
    decision: RouteDecision
    attempts: int
    rewritten: bool
    chunks: list[RetrievedChunk]
    top_vector_score: float
    graded_ok: bool
    refused: bool
    refusal_reason: str
    answer: str
    citations: list[str]
    tokens_in: int
    tokens_out: int


@dataclass
class AgentRuntime:
    """Dependencies the nodes close over (LangGraph state stays data-only)."""

    conn: psycopg.Connection
    settings: Settings
    retriever: Retriever
    answerer: Answerer


@dataclass(frozen=True)
class AgentResult:
    run_id: int
    route: str
    refused: bool
    refusal_reason: str
    answer: str
    citations: list[str]
    chunks: list[RetrievedChunk]
    top_vector_score: float
    rewritten: bool
    tokens_in: int
    tokens_out: int
    latency_ms: float = field(default=0.0)


def build_graph(runtime: AgentRuntime):
    def route_node(state: AgentState) -> AgentState:
        decision, result = plan_retrieve(
            runtime.conn, runtime.retriever, state["working_question"],
            k=state["k"], mode=state["mode"],
        )
        out: AgentState = {"decision": decision, "attempts": state.get("attempts", 0) + 1}
        if result is not None:
            out["chunks"] = result.chunks
            out["top_vector_score"] = result.top_vector_score
        return out

    def grade_node(state: AgentState) -> AgentState:
        decision = state["decision"]
        if decision.route == "out_of_scope":
            return {"graded_ok": False, "refusal_reason": decision.reason}
        chunks = state.get("chunks", [])
        top = state.get("top_vector_score", 0.0)
        bar = refusal_bar(runtime.settings, decision)
        companies_covered = all(
            any(c.company == company for c in chunks[:10]) for company in decision.companies
        )
        ok = bool(chunks) and top >= bar and companies_covered
        reason = ""
        if not ok:
            if not chunks:
                reason = "no chunks retrieved"
            elif top < bar:
                reason = f"top similarity {top:.3f} below bar {bar:.2f}"
            else:
                reason = "retrieved evidence does not cover every named company"
        return {"graded_ok": ok, "refusal_reason": reason}

    def rewrite_node(state: AgentState) -> AgentState:
        # Deterministic rewrite: anchor with company names + filing vocabulary.
        decision = state["decision"]
        anchors = " ".join(decision.companies) if decision.companies else "SEC filing"
        rewritten = f"{state['question']} {anchors} risk factors MD&A disclosure"
        logger.info("rewriting query -> %r", rewritten)
        return {"working_question": rewritten, "rewritten": True}

    def answer_node(state: AgentState) -> AgentState:
        generated = runtime.answerer.answer(state["question"], state["chunks"])
        citations = [format_citation(c) for c in state["chunks"][:6]]
        return {
            "answer": generated.text, "citations": citations, "refused": False,
            "tokens_in": generated.tokens_in, "tokens_out": generated.tokens_out,
        }

    def refuse_node(state: AgentState) -> AgentState:
        reason = state.get("refusal_reason") or "low retrieval confidence"
        return {
            "refused": True,
            "answer": f"Not covered by this corpus ({reason}). "
            "The corpus holds 10-K/10-Q filings for Apple, Microsoft, Nvidia, "
            "Tesla, Coca-Cola, and JPMorgan.",
            "citations": [],
        }

    def after_grade(state: AgentState) -> str:
        if state["graded_ok"]:
            return "answer"
        if (
            state["decision"].route != "out_of_scope"
            and state.get("attempts", 0) < MAX_ATTEMPTS
        ):
            return "rewrite"
        return "refuse"

    graph = StateGraph(AgentState)
    graph.add_node("route_retrieve", route_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("answer", answer_node)
    graph.add_node("refuse", refuse_node)
    graph.add_edge(START, "route_retrieve")
    graph.add_edge("route_retrieve", "grade")
    graph.add_conditional_edges(
        "grade", after_grade, {"answer": "answer", "rewrite": "rewrite", "refuse": "refuse"}
    )
    graph.add_edge("rewrite", "route_retrieve")
    graph.add_edge("answer", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def run_agent(
    runtime: AgentRuntime, question: str, k: int = 10, mode: str = "hybrid"
) -> AgentResult:
    started = time.perf_counter()
    app = build_graph(runtime)
    final: AgentState = app.invoke(
        {"question": question, "working_question": question, "k": k, "mode": mode}
    )
    latency_ms = (time.perf_counter() - started) * 1000

    chunks = final.get("chunks", [])
    decision = final["decision"]
    with runtime.conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO query_runs (query, route, index_version_id, mode,
                retrieved_chunk_ids, scores, top_vector_score, rewritten, refused,
                refusal_reason, answer, llm_provider, tokens_in, tokens_out, latency_ms)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING run_id
            """,
            (
                question, decision.route,
                runtime.retriever.version.index_version_id, mode,
                [c.chunk_id for c in chunks], [c.score for c in chunks],
                final.get("top_vector_score"), final.get("rewritten", False),
                final["refused"], final.get("refusal_reason") or None,
                final.get("answer"), runtime.answerer.provider,
                final.get("tokens_in", 0), final.get("tokens_out", 0), latency_ms,
            ),
        )
        run_id = cur.fetchone()[0]
    runtime.conn.commit()

    return AgentResult(
        run_id=run_id, route=decision.route, refused=final["refused"],
        refusal_reason=final.get("refusal_reason", ""), answer=final.get("answer", ""),
        citations=final.get("citations", []), chunks=chunks,
        top_vector_score=final.get("top_vector_score", 0.0),
        rewritten=final.get("rewritten", False),
        tokens_in=final.get("tokens_in", 0), tokens_out=final.get("tokens_out", 0),
        latency_ms=latency_ms,
    )
