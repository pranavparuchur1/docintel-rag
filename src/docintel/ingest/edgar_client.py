"""Rate-limited HTTP client for SEC EDGAR.

EDGAR's fair-access policy (https://www.sec.gov/os/accessing-edgar-data) requires
a User-Agent identifying a person and email, and caps clients at ~10 requests/sec.
Respecting a source's contract is a first-class requirement here, not an afterthought:
this client throttles below the cap, honors Retry-After, and backs off exponentially
with jitter on 429/5xx and transport errors. Every request is logged with status,
latency, and size.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 60.0


class EdgarError(RuntimeError):
    """A request failed after exhausting retries."""


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        max_requests_per_sec: float = 8.0,
        max_retries: int = 5,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._min_interval = 1.0 / max_requests_per_sec
        self._max_retries = max_retries
        self._last_request_at = 0.0
        self._sleep = time.sleep  # injectable for tests
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    # --- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- core ------------------------------------------------------------------
    def get(self, url: str) -> httpx.Response:
        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            self._throttle()
            started = time.monotonic()
            try:
                response = self._client.get(url)
            except httpx.TransportError as exc:
                self._last_request_at = time.monotonic()
                if attempt == self._max_retries:
                    msg = f"GET {url} failed after {attempt + 1} attempts: {exc}"
                    raise EdgarError(msg) from exc
                delay = self._backoff_delay(attempt, retry_after=None)
                logger.warning(
                    "GET %s transport error (%s) — retry %d/%d in %.1fs",
                    url, exc, attempt + 1, self._max_retries, delay,
                )
                self._sleep(delay)
                continue

            self._last_request_at = time.monotonic()
            elapsed = self._last_request_at - started
            if response.status_code in RETRYABLE_STATUS:
                last_status = response.status_code
                if attempt == self._max_retries:
                    break
                delay = self._backoff_delay(attempt, response.headers.get("Retry-After"))
                logger.warning(
                    "GET %s -> %d — retry %d/%d in %.1fs",
                    url, response.status_code, attempt + 1, self._max_retries, delay,
                )
                self._sleep(delay)
                continue

            logger.info(
                "GET %s -> %d (%.2fs, %d bytes)",
                url, response.status_code, elapsed, len(response.content),
            )
            response.raise_for_status()
            return response

        raise EdgarError(
            f"GET {url} still returning {last_status} after {self._max_retries + 1} attempts"
        )

    def get_json(self, url: str) -> dict:
        return self.get(url).json()

    def download(self, url: str, dest: Path) -> tuple[str, int]:
        """Fetch url to dest unmodified; return (sha256, size_bytes)."""
        response = self.get(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return hashlib.sha256(response.content).hexdigest(), len(response.content)

    # --- politeness ------------------------------------------------------------
    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_request_at)
        if wait > 0:
            self._sleep(wait)

    def _backoff_delay(self, attempt: int, retry_after: str | None) -> float:
        # The server's own instruction wins over our schedule.
        if retry_after:
            try:
                return min(MAX_BACKOFF_SECONDS, float(retry_after))
            except ValueError:
                pass
        # 1s, 2s, 4s, 8s, ... capped, plus jitter so parallel runs don't sync up.
        return min(MAX_BACKOFF_SECONDS, float(2**attempt)) + random.uniform(0, 0.5)
