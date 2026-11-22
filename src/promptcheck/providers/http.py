"""Minimal JSON-over-HTTP client with retries, built on ``urllib``."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

from promptcheck import __version__
from promptcheck.errors import ProviderError, ProviderHTTPError

USER_AGENT = f"promptcheck/{__version__} (+https://github.com/Michealjunior/promptcheck)"


def is_retryable_status(status: int) -> bool:
    """429 (rate limited), 408 (request timeout) and every 5xx are worth retrying."""
    return status in (408, 429) or 500 <= status <= 599


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff: ``backoff * 2**attempt`` seconds, capped at ``max_delay``."""

    retries: int = 3
    backoff: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return max(0.0, min(retry_after, self.max_delay))
        delay = min(self.backoff * (2**attempt), self.max_delay)
        if self.jitter:
            delay *= 1 + random.uniform(0, 0.25)
        return min(delay, self.max_delay)


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header given in seconds or as an HTTP date."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, moment.timestamp() - time.time())


def redact(text: str, secrets: Iterable[str | None]) -> str:
    """Replace any secret value that leaked into ``text`` with ``***``."""
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***")
    return text


def error_message(body: bytes) -> str:
    """Extract a readable message from an API error body (OpenAI or Anthropic shape)."""
    text = body.decode("utf-8", errors="replace").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:300] or "no response body"
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            kind = err.get("type")
            return f"{err['message']} ({kind})" if isinstance(kind, str) and kind else err["message"]
        if isinstance(err, str):
            return err
        if isinstance(data.get("message"), str):
            return data["message"]
    return text[:300]


Opener = Callable[..., Any]


def post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    policy: RetryPolicy | None = None,
    secrets: Iterable[str | None] = (),
    sleep: Callable[[float], None] = time.sleep,
    opener: Opener | None = None,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON response.

    Retries 408/429/5xx responses, connection errors and timeouts with exponential
    backoff (honouring ``Retry-After``). Other HTTP errors raise immediately.
    Secrets are scrubbed from every error message.
    """
    policy = policy or RetryPolicy()
    open_url = opener or urllib.request.urlopen
    secrets = tuple(secrets)
    data = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
        **(headers or {}),
    }
    attempt = 0
    while True:
        request = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
        try:
            with open_url(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                body = exc.read() or b""
            except Exception:  # pragma: no cover - defensive: broken error stream
                body = b""
            if is_retryable_status(status) and attempt < policy.retries:
                retry_after = parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                sleep(policy.delay(attempt, retry_after))
                attempt += 1
                continue
            raise ProviderHTTPError(
                status, redact(error_message(body), secrets), attempts=attempt + 1
            ) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < policy.retries:
                sleep(policy.delay(attempt))
                attempt += 1
                continue
            reason = getattr(exc, "reason", exc)
            if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
                what = f"timed out after {timeout:g}s"
            else:
                what = f"connection failed: {reason}"
            tries = f" (after {attempt + 1} attempts)" if attempt else ""
            raise ProviderError(redact(f"{what}{tries}", secrets)) from None
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderError("the provider returned a response that is not JSON") from None
        if not isinstance(decoded, dict):
            raise ProviderError("the provider returned unexpected JSON (expected an object)")
        return decoded
