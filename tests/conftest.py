from __future__ import annotations

import io
import json
import textwrap
import threading
import urllib.error
from collections.abc import Callable
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from promptcheck.providers.base import Completion, CompletionRequest, Provider
from promptcheck.suite import Suite, load_suite

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_SUITE = ROOT / "examples" / "support-triage.yaml"
EXAMPLE_RECORDINGS = ROOT / "examples" / "recordings.json"


@pytest.fixture
def write_suite(tmp_path: Path) -> Callable[[str], Path]:
    """Write dedented YAML to a temporary suite file and return its path."""

    def write(text: str, name: str = "suite.yaml") -> Path:
        path = tmp_path / name
        path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
        return path

    return write


@pytest.fixture
def make_suite(write_suite: Callable[[str], Path]) -> Callable[[str], Suite]:
    def make(text: str) -> Suite:
        return load_suite(write_suite(text))

    return make


class FakeProvider(Provider):
    """Returns canned outputs keyed by case id (a list cycles per sample)."""

    name = "fake"

    def __init__(self, outputs: dict[str, Any], *, raise_for: dict[str, Exception] | None = None) -> None:
        super().__init__()
        self.outputs = outputs
        self.raise_for = raise_for or {}
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> Completion:
        self.requests.append(request)
        if request.case_id in self.raise_for:
            raise self.raise_for[request.case_id]
        value = self.outputs[request.case_id]
        if isinstance(value, list):
            value = value[request.sample % len(value)]
        if isinstance(value, Completion):
            return value
        return Completion(text=value, model="fake-model")


class FakeResponse:
    """Stands in for the object returned by ``urllib.request.urlopen``."""

    def __init__(self, payload: Any, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
        self._position = 0

    def read(self, amount: int | None = None) -> bytes:
        end = len(self._body) if amount is None else self._position + amount
        chunk = self._body[self._position : end]
        self._position += len(chunk)
        return chunk

    read1 = read

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def http_error(
    status: int, payload: Any = None, headers: dict[str, str] | None = None
) -> urllib.error.HTTPError:
    message = Message()
    for key, value in (headers or {}).items():
        message[key] = value
    body = json.dumps(payload if payload is not None else {"error": {"message": f"status {status}"}})
    return urllib.error.HTTPError(
        "https://example.test", status, f"HTTP {status}", message, io.BytesIO(body.encode())
    )


class FakeOpener:
    """Callable replacement for urlopen that replays a scripted list of responses or errors."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.requests: list[Any] = []
        self.timeouts: list[float] = []

    def __call__(self, request: Any, timeout: float | None = None) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if not self.script:
            raise AssertionError("unexpected extra HTTP request")
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step if isinstance(step, FakeResponse) else FakeResponse(step)

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].data.decode())

    def header(self, name: str, index: int = -1) -> str | None:
        # urllib normalises header names with str.capitalize()
        return self.requests[index].get_header(name.capitalize())


class StubServer:
    """A real HTTP server on 127.0.0.1 for end-to-end tests. Never talks to a real API.

    Set ``respond`` to a function ``(handler, path, body) -> None`` that writes the reply.
    Every request is recorded in ``requests`` as ``(method, path, headers, body)``.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, str], Any]] = []
        self.respond: Callable[[BaseHTTPRequestHandler, str, Any], None] = self.reply_json
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except json.JSONDecodeError:
                    body = raw
                stub.requests.append((self.command, self.path, dict(self.headers), body))
                stub.respond(self, self.path, body)

            do_POST = _handle
            do_GET = _handle

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.handle_error = lambda request, address: None  # clients may hang up on purpose
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @staticmethod
    def reply_json(
        handler: BaseHTTPRequestHandler, path: str, body: Any, payload: Any = None, status: int = 200
    ) -> None:
        data = json.dumps(payload if payload is not None else {"ok": True}).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub_server():
    servers: list[StubServer] = []

    def start() -> StubServer:
        server = StubServer()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()
