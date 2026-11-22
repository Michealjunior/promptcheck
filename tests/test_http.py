from __future__ import annotations

import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from promptcheck.errors import ProviderError, ProviderHTTPError
from promptcheck.providers.http import (
    RetryPolicy,
    error_message,
    is_retryable_status,
    parse_retry_after,
    post_json,
    redact,
)

from .conftest import FakeOpener, FakeResponse, http_error

NO_JITTER = RetryPolicy(retries=3, backoff=1.0, max_delay=30.0, jitter=False)


def call(opener, policy=NO_JITTER, **kwargs):
    sleeps: list[float] = []
    result = post_json(
        "https://api.example.test/v1/thing",
        {"hello": "world"},
        policy=policy,
        sleep=sleeps.append,
        opener=opener,
        **kwargs,
    )
    return result, sleeps


def test_success_sends_json_with_headers():
    opener = FakeOpener({"ok": True})
    result, sleeps = call(opener, headers={"X-Test": "1"}, timeout=7)
    assert result == {"ok": True}
    assert sleeps == []
    assert opener.body() == {"hello": "world"}
    assert opener.header("Content-Type") == "application/json"
    assert opener.header("X-test") == "1"
    assert opener.header("User-Agent").startswith("promptcheck/")
    assert opener.timeouts == [7]
    assert opener.requests[0].get_method() == "POST"


@pytest.mark.parametrize("status", [429, 500, 502, 503, 529, 408])
def test_retries_transient_statuses_with_exponential_backoff(status):
    opener = FakeOpener(http_error(status), http_error(status), http_error(status), {"ok": 1})
    result, sleeps = call(opener)
    assert result == {"ok": 1}
    assert sleeps == [1.0, 2.0, 4.0]
    assert len(opener.requests) == 4


def test_gives_up_after_the_retry_budget():
    opener = FakeOpener(*(http_error(503, {"error": {"message": "overloaded"}}) for _ in range(3)))
    with pytest.raises(ProviderHTTPError) as info:
        call(opener, policy=RetryPolicy(retries=2, jitter=False))
    assert info.value.status == 503
    assert info.value.attempts == 3
    assert str(info.value) == "HTTP 503 after 3 attempts: overloaded"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(status):
    opener = FakeOpener(http_error(status, {"error": {"message": "bad request", "type": "invalid"}}))
    with pytest.raises(ProviderHTTPError, match=rf"HTTP {status}: bad request \(invalid\)"):
        call(opener)
    assert len(opener.requests) == 1


def test_retry_after_header_is_honoured_and_capped():
    opener = FakeOpener(
        http_error(429, headers={"Retry-After": "7"}),
        http_error(429, headers={"Retry-After": "999"}),
        {"ok": 1},
    )
    _, sleeps = call(opener)
    assert sleeps == [7.0, 30.0]


def test_network_errors_and_timeouts_are_retried():
    opener = FakeOpener(
        urllib.error.URLError(ConnectionRefusedError("refused")), TimeoutError("slow"), {"ok": 1}
    )
    result, sleeps = call(opener)
    assert result == {"ok": 1}
    assert sleeps == [1.0, 2.0]


def test_timeout_message_after_retries():
    opener = FakeOpener(TimeoutError(), TimeoutError())
    with pytest.raises(ProviderError, match=r"timed out after 5s \(after 2 attempts\)"):
        call(opener, policy=RetryPolicy(retries=1, jitter=False), timeout=5)


def test_connection_failure_message():
    opener = FakeOpener(urllib.error.URLError("Name or service not known"))
    with pytest.raises(ProviderError, match="connection failed: Name or service not known"):
        call(opener, policy=RetryPolicy(retries=0))


def test_non_json_response_is_a_provider_error():
    with pytest.raises(ProviderError, match="not JSON"):
        call(FakeOpener(FakeResponse(b"<html>gateway</html>")))


def test_secrets_are_redacted_from_errors():
    secret = "sk-test-SECRET-123456"
    opener = FakeOpener(http_error(401, {"error": {"message": f"Incorrect API key provided: {secret}"}}))
    with pytest.raises(ProviderHTTPError) as info:
        call(opener, secrets=[secret])
    assert secret not in str(info.value)
    assert "***" in str(info.value)


def test_backoff_policy():
    policy = RetryPolicy(backoff=0.5, max_delay=3, jitter=False)
    assert [policy.delay(i) for i in range(5)] == [0.5, 1.0, 2.0, 3.0, 3.0]
    jittery = RetryPolicy(backoff=1, jitter=True)
    assert all(1.0 <= jittery.delay(0) <= 1.25 for _ in range(50))


def test_parse_retry_after():
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    assert 25 <= parse_retry_after(future) <= 31


def test_helpers():
    assert is_retryable_status(429) and is_retryable_status(500) and not is_retryable_status(404)
    assert redact("key=abcd1234", ["abcd1234", None, ""]) == "key=***"
    assert error_message(b'{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}') == (
        "Overloaded (overloaded_error)"
    )
    assert error_message(b'{"error":"plain"}') == "plain"
    assert error_message(b"") == "no response body"


def test_default_opener_is_urllib(monkeypatch):
    opener = FakeOpener({"via": "urlopen"})
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    assert post_json("https://x.test", {}, policy=NO_JITTER) == {"via": "urlopen"}
