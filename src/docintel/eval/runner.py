"""Eval runner: golden set × (index version, retrieval mode) -> judged metrics
and a markdown report under docs/eval/."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from docintel.embed.providers import EmbeddingProvider
from docintel.embed.service import IndexVersion
from docintel.eval.golden import GoldenQuestion, relevant_universe_size
from docintel.eval.metrics import (
    QueryJudgment,
    mrr_at_10,
    ndcg_at_10,
    percentile,
    recall_at,
    refusal_precision_recall,
)
from docintel.retrieve.service import Retriever

logger = logging.getLogger(__name__)

RECALL_KS = (1, 5, 10)


@dataclass
class EvalRun:
    version: IndexVersion
    mode: str
    threshold: float
    judgments: list[QueryJudgment]

    @property
    def label(self) -> str:
        return f"v{self.version.index_version_id} {self.version.strategy} / {self.mode}"

    def metrics(self) -> dict[str, float]:
        latencies = [j.latency_ms for j in self.judgments]
        precision, recall = refusal_precision_recall(self.judgments)
        return {
            "recall@1": recall_at(self.judgments, 1),
            "recall@5": recall_at(self.judgments, 5),
            "recall@10": recall_at(self.judgments, 10),
            "mrr@10": mrr_at_10(self.judgments),
            "ndcg@10": ndcg_at_10(self.judgments),
            "refusal_precision": precision,
            "refusal_recall": recall,
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
        }


def judge_question(
    conn: psycopg.Connection,
    retriever: Retriever,
    question: GoldenQuestion,
    mode: str,
    threshold: float,
    k: int = 10,
    settings=None,
) -> QueryJudgment:
    if mode == "agent":
        # measure exactly what the agent retrieves: routing + fan-out + the
        # entity guard and asymmetric refusal bars
        from docintel.agent.plan import plan_retrieve, refusal_bar
        from docintel.retrieve.service import RetrievalResult

        decision, result = plan_retrieve(conn, retriever, question.question, k=k)
        if result is None:
            result = RetrievalResult(chunks=[], top_vector_score=0.0, latency_ms=0.0)
            refused = True
        else:
            refused = result.top_vector_score < refusal_bar(settings, decision)
    else:
        result = retriever.search(question.question, k=k, mode=mode)
        refused = result.top_vector_score < threshold
    relevance = [
        any(spec.matches(chunk) for spec in question.expected) for chunk in result.chunks
    ]
    if question.require == "any":
        # alternate sources: any single matching spec fully answers the question
        specs_total = 1 if question.expected else 0
        specs_hit_at = {
            kk: int(
                any(
                    spec.matches(chunk)
                    for spec in question.expected
                    for chunk in result.chunks[:kk]
                )
            )
            for kk in RECALL_KS
        }
    else:
        specs_total = len(question.expected)
        specs_hit_at = {
            kk: sum(
                1
                for spec in question.expected
                if any(spec.matches(chunk) for chunk in result.chunks[:kk])
            )
            for kk in RECALL_KS
        }
    first = next((i + 1 for i, rel in enumerate(relevance) if rel), None)
    universe = (
        relevant_universe_size(conn, retriever.version, question.expected)
        if question.expected
        else 0
    )
    return QueryJudgment(
        question_id=question.id,
        question_type=question.type,
        should_refuse=question.refuse,
        refused=refused,
        specs_total=specs_total,
        specs_hit_at=specs_hit_at,
        first_relevant_rank=first,
        relevance=relevance,
        relevant_universe=universe,
        latency_ms=result.latency_ms,
        top_vector_score=result.top_vector_score,
    )


def run_eval(
    conn: psycopg.Connection,
    provider: EmbeddingProvider,
    version: IndexVersion,
    golden: list[GoldenQuestion],
    mode: str,
    threshold: float,
    settings=None,
) -> EvalRun:
    retriever = Retriever(conn, provider, version)
    judgments = []
    for question in golden:
        judgment = judge_question(conn, retriever, question, mode, threshold, settings=settings)
        judgments.append(judgment)
        logger.info(
            "%s %s %s: first_rel=%s top_vec=%.3f%s",
            version.index_version_id, mode, question.id,
            judgment.first_relevant_rank, judgment.top_vector_score,
            " REFUSED" if judgment.refused else "",
        )
    return EvalRun(version=version, mode=mode, threshold=threshold, judgments=judgments)


def threshold_sweep(run: EvalRun, thresholds: list[float]) -> list[tuple[float, float, float]]:
    """Refusal (threshold, precision, recall) using the recorded top-1 vector
    scores — no re-retrieval needed."""
    out = []
    for t in thresholds:
        refused = [j for j in run.judgments if j.top_vector_score < t]
        should = [j for j in run.judgments if j.should_refuse]
        precision = (
            sum(1 for j in refused if j.should_refuse) / len(refused) if refused else 1.0
        )
        recall = sum(1 for j in should if j.top_vector_score < t) / len(should) if should else 1.0
        out.append((t, precision, recall))
    return out


def _metrics_table(runs: list[EvalRun]) -> list[str]:
    cols = [
        "recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10",
        "refusal_precision", "refusal_recall", "p50_ms", "p95_ms",
    ]
    lines = ["| configuration | " + " | ".join(cols) + " |"]
    lines.append("|---" * (len(cols) + 1) + "|")
    for run in runs:
        m = run.metrics()
        cells = [f"{m[c]:.2f}" if not c.endswith("_ms") else f"{m[c]:.0f}" for c in cols]
        lines.append(f"| {run.label} | " + " | ".join(cells) + " |")
    return lines


def write_report(
    runs: list[EvalRun], golden: list[GoldenQuestion], out_dir: Path, name: str | None = None
) -> Path:
    now = datetime.now(UTC)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name or f"eval_{now:%Y-%m-%d_%H%M}.md")
    verified = sum(1 for q in golden if q.verified)
    in_corpus = [q for q in golden if not q.refuse]
    lines = [
        f"# Retrieval evaluation — {now:%Y-%m-%d %H:%M} UTC",
        "",
        f"- golden set: **{len(golden)} questions** ({len(in_corpus)} in-corpus, "
        f"{len(golden) - len(in_corpus)} out-of-corpus), "
        f"**{verified}/{len(golden)} human-verified**",
        f"- embedding model: `{runs[0].version.embedding_model}` · "
        f"refusal threshold (top-1 cosine): {runs[0].threshold}",
        f"- caveat: with {len(in_corpus)} in-corpus questions one question moves recall by "
        f"~{1 / max(len(in_corpus), 1):.3f}; two decimals are reported, differences smaller "
        "than ~0.05 are noise.",
        "",
        "## Results",
        "",
        *_metrics_table(runs),
        "",
        "recall@k is spec coverage (a cross-company question counts fully only when every "
        "expected source is retrieved); definitions in `src/docintel/eval/metrics.py`.",
        "",
        "## Per-type recall@5",
        "",
    ]
    types = sorted({q.type for q in in_corpus})
    lines.append("| configuration | " + " | ".join(types) + " |")
    lines.append("|---" * (len(types) + 1) + "|")
    for run in runs:
        cells = []
        for qtype in types:
            js = [j for j in run.judgments if j.question_type == qtype and not j.should_refuse]
            cells.append(f"{recall_at(js, 5):.2f}" if js else "—")
        lines.append(f"| {run.label} | " + " | ".join(cells) + " |")

    lines += ["", "## Refusal threshold sweep (top-1 cosine)", ""]
    lines.append("| threshold | " + " | ".join(r.label for r in runs) + " |")
    lines.append("|---" * (len(runs) + 1) + "|")
    for t in [round(0.40 + 0.02 * s, 2) for s in range(16)]:
        row = [f"{t:.2f}"]
        for run in runs:
            sweep = threshold_sweep(run, [t])[0]
            row.append(f"P={sweep[1]:.2f} R={sweep[2]:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Cost per query",
        "",
        "Local embeddings + local Postgres: **$0.00 marginal cost per query** (measured "
        "configuration). For reference, the same query embedded with OpenAI "
        "text-embedding-3-small would cost ≈ $0.0000004 at ~20 query tokens ($0.02/1M tokens); "
        "retrieval itself stays free either way. LLM generation cost lands in Phase 5.",
    ]

    lines += ["", "## Misses (no relevant chunk in top 10)", ""]
    for run in runs:
        missed = [
            j.question_id
            for j in run.judgments
            if not j.should_refuse and j.first_relevant_rank is None
        ]
        lines.append(f"- {run.label}: {', '.join(missed) if missed else 'none'}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
