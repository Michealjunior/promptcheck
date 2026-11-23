"""Provider interface shared by live and replay backends."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from promptcheck.providers.http import redact
from promptcheck.suite import request_fingerprint


@dataclass(frozen=True)
class CompletionRequest:
    """Everything a provider needs to produce one sample for one case."""

    case_id: str
    sample: int
    model: str | None
    system: str | None
    messages: tuple[dict[str, str], ...]
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float = 60.0

    def fingerprint(self) -> str:
        return request_fingerprint(self.model, self.system, list(self.messages))


@dataclass
class Completion:
    """A provider's answer to one request."""

    text: str
    model: str | None = None
    refused: bool = False
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    warnings: list[str] = field(default_factory=list)


class Provider(ABC):
    """A backend that turns a :class:`CompletionRequest` into a :class:`Completion`."""

    name: str = "provider"

    def __init__(self) -> None:
        self._notices: list[str] = []
        self._notice_lock = threading.Lock()

    @property
    def label(self) -> str:
        """Short human-readable description used in report headers."""
        return self.name

    @property
    def notices(self) -> list[str]:
        """Run-level notes (each reported once), e.g. ignored settings."""
        with self._notice_lock:
            return list(self._notices)

    def notice(self, message: str) -> None:
        with self._notice_lock:
            if message not in self._notices:
                self._notices.append(message)

    @abstractmethod
    def complete(self, request: CompletionRequest) -> Completion:
        """Return a completion or raise :class:`promptcheck.errors.ProviderError`."""


def scrub_secret(completion: Completion, secret: str | None) -> Completion:
    """Remove an API key that a misbehaving server echoed into the output itself."""
    if not secret:
        return completion
    cleaned = redact(completion.text, (secret,))
    if cleaned != completion.text:
        completion.text = cleaned
        completion.warnings.append("the response contained the API key; it was replaced with ***")
    return completion
