"""Minimal JSON-over-HTTP client with retries, built on ``urllib``."""

from __future__ import annotations

import contextlib
import http.client
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

from promptcheck import __version__
from promptcheck.errors import ProviderError, ProviderHTTPError

USER_AGENT = f"promptcheck/{__version__} (+https://github.com/StarJnrDev/promptcheck)"
MAX_RETRIES = 10
"""Upper bound for ``--retries``."""

# Network failures worth retrying: URLError, timeouts and resets are OSErrors; a body cut
# short (IncompleteRead) or a dropped connection (RemoteDisconnected) are HTTPExceptions.
TRANSIENT_ERRORS = (OSError, http.client.HTTPException)


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects.

    urllib would turn a redirected POST into a GET without its body (301/302/303) and
    forward the Authorization header to whatever host the Location names. Neither is
    acceptable for an API call, so a redirect becomes a clear error instead.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirects)


def urlopen(request: urllib.request.Request, timeout: float) -> Any:
    """Open ``request`` without following redirects. Tests replace this function."""
    return _OPENER.open(request, timeout=timeout)


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
    """Replace any secret value that leaked into ``text`` with ``***`` (ignoring case)."""
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = re.sub(re.escape(secret), "***", text, flags=re.IGNORECASE)
    return text


def _set_socket_timeout(response: Any, seconds: float) -> None:
    # Best effort: shorten the socket timeout so a stalled read cannot outlive the deadline.
    sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    if sock is not None:
        with contextlib.suppress(OSError, ValueError):
            sock.settimeout(max(seconds, 0.001))


def read_body(response: Any, deadline: float) -> bytes:
    """Read the whole response body, raising TimeoutError once ``deadline`` (monotonic) passes.

    A socket timeout only bounds each individual read, so a server that trickles a
    byte at a time could otherwise hold a request open far longer than ``--timeout``.
    """
    chunks: list[bytes] = []
    read = getattr(response, "read1", None) or response.read
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("deadline exceeded while reading the response")
        _set_socket_timeout(response, remaining)
        chunk = read(65536)
        if not chunk:
            body = b"".join(chunks)
            missing = getattr(response, "length", None)
            if isinstance(missing, int) and missing > 0:
                # The server closed the connection before sending the promised Content-Length.
                raise http.client.IncompleteRead(body, missing)
            return body
        chunks.append(chunk)


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

    ``timeout`` is a deadline for each attempt as a whole (connecting, waiting and
    reading the body), not just for individual socket reads. 408/429/5xx responses,
    connection errors, truncated bodies and timeouts are retried with exponential
    backoff (honouring ``Retry-After``). Redirects are never followed. Other HTTP
    errors raise immediately. Secrets are scrubbed from every error message.
    """
    policy = policy or RetryPolicy()
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
        open_url = opener or urlopen
        request = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
        deadline = time.monotonic() + timeout
        try:
            with open_url(request, timeout=timeout) as response:
                raw = read_body(response, deadline)
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                body = exc.read() or b""
            except Exception:  # pragma: no cover - defensive: broken error stream
                body = b""
            if 300 <= status < 400:
                location = exc.headers.get("Location", "another address") if exc.headers else "elsewhere"
                raise ProviderHTTPError(
                    status,
                    redact(
                        f"the server redirected to {location}; promptcheck does not follow redirects "
                        "for API requests (the request body would be dropped and the API key could be "
                        "sent to another host). Set base_url to the final address.",
                        secrets,
                    ),
                ) from None
            if is_retryable_status(status) and attempt < policy.retries:
                retry_after = parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                sleep(policy.delay(attempt, retry_after))
                attempt += 1
                continue
            raise ProviderHTTPError(
                status, redact(error_message(body), secrets), attempts=attempt + 1
            ) from None
        except TRANSIENT_ERRORS as exc:
            if attempt < policy.retries:
                sleep(policy.delay(attempt))
                attempt += 1
                continue
            reason = getattr(exc, "reason", exc)
            if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
                what = f"timed out after {timeout:g}s"
            elif isinstance(exc, http.client.IncompleteRead):
                what = "the connection closed before the whole response arrived"
            elif isinstance(exc, http.client.HTTPException) and not isinstance(exc, OSError):
                what = f"invalid HTTP response: {type(exc).__name__}"
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
