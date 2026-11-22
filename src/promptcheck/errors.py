"""Exception types shared across promptcheck."""

from __future__ import annotations


class PromptcheckError(Exception):
    """Base class for every error promptcheck raises on purpose."""


class SuiteError(PromptcheckError):
    """A suite file is missing, unreadable or invalid.

    The message always names the file and, when relevant, the case id so the
    author can jump straight to the problem.
    """


class ConfigError(PromptcheckError):
    """Command-line or provider configuration is unusable (for example a missing API key)."""


class ProviderError(PromptcheckError):
    """A provider could not produce a completion for one request."""


class ProviderHTTPError(ProviderError):
    """The provider answered with a non-success HTTP status."""

    def __init__(self, status: int, message: str, attempts: int = 1) -> None:
        self.status = status
        self.attempts = attempts
        suffix = f" after {attempts} attempts" if attempts > 1 else ""
        super().__init__(f"HTTP {status}{suffix}: {message}")
