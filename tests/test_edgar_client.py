"""EDGAR client contract: identify yourself, throttle, back off, give up loudly."""

import httpx
import pytest

from docintel.ingest.edgar_client import EdgarClient, EdgarError

UA = "Jane Doe jane@example.com"


def make_client(handler, **kwargs) -> EdgarClient:
    client = EdgarClient(UA, transport=httpx.MockTransport(handler), **kwargs)
    client._sleep = lambda _s: None  # never actually sleep in unit tests
    return client


def test_user_agent_header_is_sent():
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["User-Agent"]
        return httpx.Response(200, text="ok")

    make_client(handler).get("https://data.sec.gov/x")
    assert seen["ua"] == UA


def test_retries_on_429_then_succeeds():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429)
        return httpx.Response(200, text="ok")

    response = make_client(handler).get("https://data.sec.gov/x")
    assert response.status_code == 200
    assert len(attempts) == 3


def test_gives_up_after_max_retries():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(429)

    with pytest.raises(EdgarError, match="429"):
        make_client(handler, max_retries=2).get("https://data.sec.gov/x")
    assert len(attempts) == 3  # initial try + 2 retries


def test_backoff_grows_exponentially_and_is_capped():
    client = make_client(lambda r: httpx.Response(200))
    delays = [client._backoff_delay(attempt, retry_after=None) for attempt in range(6)]
    for earlier, later in zip(delays, delays[1:5], strict=False):
        assert later > earlier
    assert client._backoff_delay(30, retry_after=None) <= 60.5  # capped + jitter


def test_retry_after_header_wins_over_schedule():
    client = make_client(lambda r: httpx.Response(200))
    assert client._backoff_delay(0, retry_after="7") == 7.0


def test_throttle_spaces_out_requests():
    slept = []
    client = EdgarClient(
        UA,
        max_requests_per_sec=5.0,  # 200ms minimum spacing
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="ok")),
    )
    client._sleep = slept.append
    client.get("https://data.sec.gov/a")
    client.get("https://data.sec.gov/b")
    assert len(slept) == 1  # only the second request needed to wait
    assert 0 < slept[0] <= 0.2


def test_non_retryable_4xx_raises_immediately():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        make_client(handler).get("https://data.sec.gov/missing")
    assert len(attempts) == 1
