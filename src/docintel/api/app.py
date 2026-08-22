"""FastAPI serving layer: POST /query, GET /health, GET /metrics.

Auth-agnostic but not naive: strict input validation (pydantic), a per-IP
sliding-window rate limit on /query, and no secrets in responses. The rate
limiter is in-process (documented limitation: per-worker, resets on restart —
a shared store is the production upgrade, not more code here).

Dependencies (get_conn, rate_limit) are module-level so tests can override
them through FastAPI's standard dependency_overrides mechanism.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from docintel import __version__, db
from docintel.agent.graph import AgentRuntime, run_agent
from docintel.agent.llm import get_answerer
from docintel.config import Settings, get_settings
from docintel.embed.providers import get_provider
from docintel.embed.service import get_index_version, get_latest_ready_version
from docintel.retrieve.service import Retriever

RATE_LIMIT_MAX = 30  # /query requests per window per client IP
RATE_LIMIT_WINDOW_S = 60.0


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    index_version: int | None = Field(default=None, ge=1)
    mode: str = Field(default="hybrid", pattern="^(vector|hybrid)$")
    k: int = Field(default=10, ge=1, le=20)


class QueryResponse(BaseModel):
    run_id: int
    route: str
    refused: bool
    answer: str
    citations: list[str]
    index_version: int
    top_vector_score: float
    latency_ms: float


def get_conn(request: Request):
    with db.connect(request.app.state.settings) as conn:
        yield conn


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    window = request.app.state.hits[ip]
    now = time.monotonic()
    while window and now - window[0] > RATE_LIMIT_WINDOW_S:
        window.popleft()
    if len(window) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429, detail="rate limit exceeded",
            headers={"Retry-After": str(int(RATE_LIMIT_WINDOW_S))},
        )
    window.append(now)


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings or get_settings()
        # embedding provider loads once per process, not per request
        app.state.provider = get_provider(app.state.settings)
        app.state.answerer = get_answerer(app.state.settings)
        app.state.hits = defaultdict(deque)
        yield

    app = FastAPI(title="docintel", version=__version__, lifespan=lifespan)

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
    def query(body: QueryRequest, request: Request, conn=Depends(get_conn)) -> QueryResponse:
        try:
            version = (
                get_index_version(conn, body.index_version)
                if body.index_version
                else get_latest_ready_version(conn)
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        runtime = AgentRuntime(
            conn=conn, settings=request.app.state.settings,
            retriever=Retriever(conn, request.app.state.provider, version),
            answerer=request.app.state.answerer,
        )
        result = run_agent(runtime, body.question, k=body.k, mode=body.mode)
        return QueryResponse(
            run_id=result.run_id, route=result.route, refused=result.refused,
            answer=result.answer, citations=result.citations,
            index_version=version.index_version_id,
            top_vector_score=result.top_vector_score, latency_ms=result.latency_ms,
        )

    @app.get("/health")
    def health(conn=Depends(get_conn)) -> dict:
        info = db.health(conn)
        version = get_latest_ready_version(conn)
        return {
            "status": "ok",
            "postgres": info["postgres"],
            "pgvector": info["pgvector"],
            "serving_index_version": version.index_version_id,
        }

    @app.get("/metrics")
    def metrics(conn=Depends(get_conn)) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*),
                       coalesce(avg(refused::int), 0),
                       coalesce(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms), 0),
                       coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), 0)
                FROM query_runs
                """
            )
            total, refusal_rate, p50, p95 = cur.fetchone()
        version = get_latest_ready_version(conn)
        return {
            "query_count": total,
            "refusal_rate": round(float(refusal_rate), 4),
            "latency_p50_ms": round(float(p50), 1),
            "latency_p95_ms": round(float(p95), 1),
            "serving_index_version": version.index_version_id,
        }

    return app
