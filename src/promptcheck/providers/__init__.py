"""Provider backends and the factory that picks one from suite + CLI settings."""

from __future__ import annotations

from pathlib import Path

from promptcheck.errors import ConfigError
from promptcheck.providers.anthropic import AnthropicProvider
from promptcheck.providers.base import Completion, CompletionRequest, Provider
from promptcheck.providers.http import RetryPolicy
from promptcheck.providers.openai import PRESETS, OpenAICompatibleProvider
from promptcheck.providers.replay import RecordingProvider, ReplayProvider
from promptcheck.suite import ProviderSettings

__all__ = [
    "AnthropicProvider",
    "Completion",
    "CompletionRequest",
    "OpenAICompatibleProvider",
    "Provider",
    "RecordingProvider",
    "ReplayProvider",
    "RetryPolicy",
    "create_provider",
]


def create_provider(
    settings: ProviderSettings,
    *,
    name: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    replay: str | Path | None = None,
    policy: RetryPolicy | None = None,
    environ: dict[str, str] | None = None,
) -> Provider:
    """Build the provider for a run. Command-line values override the suite's defaults."""
    chosen = name or ("replay" if replay else settings.name)
    if chosen == "replay":
        if not replay:
            raise ConfigError("--provider replay needs --replay PATH pointing at a recording file")
        return ReplayProvider(replay)
    if replay:
        raise ConfigError(f"--replay only works with --provider replay (got --provider {chosen})")
    # A base_url in the suite belongs to the suite's provider; don't leak it into another one.
    same_provider = chosen == settings.name
    # An explicit base URL (even an empty one) is never replaced by a default: the
    # provider validates it, so a typo cannot silently send the API key elsewhere.
    resolved_url = base_url if base_url is not None else (settings.base_url if same_provider else None)
    key_env = settings.api_key_env if same_provider else None
    resolved_model = model or settings.model
    if chosen == "anthropic":
        return AnthropicProvider(
            model=resolved_model,
            base_url=resolved_url,
            api_key_env=key_env,
            api_version=settings.anthropic_version,
            policy=policy,
            environ=environ,
        )
    if chosen in PRESETS:
        return OpenAICompatibleProvider(
            chosen,
            model=resolved_model,
            base_url=resolved_url,
            api_key_env=key_env,
            policy=policy,
            environ=environ,
        )
    raise ConfigError(f"unknown provider {chosen!r}")
