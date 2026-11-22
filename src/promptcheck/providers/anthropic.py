"""Anthropic Messages API provider (``POST /v1/messages``)."""

from __future__ import annotations

import os
from typing import Any

from promptcheck.errors import ConfigError, ProviderError
from promptcheck.providers import http
from promptcheck.providers.base import Completion, CompletionRequest, Provider

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_API_VERSION = "2026-10-21"
LEGACY_API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
"""The Messages API requires max_tokens. Thinking tokens count towards it on newer models."""


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        api_version: str | None = None,
        policy: http.RetryPolicy | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        env = os.environ if environ is None else environ
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_version = api_version or DEFAULT_API_VERSION
        self.policy = policy or http.RetryPolicy()
        self.api_key_env = api_key_env or "ANTHROPIC_API_KEY"
        self._api_key = env.get(self.api_key_env)
        if not self._api_key:
            raise ConfigError(
                f"{self.api_key_env} is not set. Export it (export {self.api_key_env}=...) "
                "or run offline with --provider replay --replay recordings.json"
            )

    @property
    def label(self) -> str:
        if self.base_url != DEFAULT_BASE_URL:
            return f"anthropic ({self.base_url})"
        return "anthropic"

    @property
    def url(self) -> str:
        return f"{self.base_url}/v1/messages"

    @property
    def sends_sampling_params(self) -> bool:
        # API version 2026-10-21 rejects temperature/top_p/top_k on every model.
        return self.api_version == LEGACY_API_VERSION

    def headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key or "", "anthropic-version": self.api_version}

    def build_payload(self, request: CompletionRequest) -> dict[str, Any]:
        model = request.model or self.model
        if not model:
            raise ProviderError("no model configured; set defaults.model in the suite or pass --model")
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": [dict(m) for m in request.messages],
        }
        if request.system:
            payload["system"] = request.system
        if request.temperature is not None:
            if self.sends_sampling_params:
                payload["temperature"] = request.temperature
            else:
                self.notice(
                    f"temperature is not sent to Anthropic: API version {self.api_version} only "
                    f"samples with the model default (set anthropic_version: {LEGACY_API_VERSION} "
                    "in defaults to send it to older models)"
                )
        return payload

    def parse_response(self, data: dict[str, Any]) -> Completion:
        content = data.get("content")
        if not isinstance(content, list):
            raise ProviderError("unexpected response: no 'content' in the message")
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        stop = data.get("stop_reason")
        warnings = []
        if stop == "max_tokens":
            warnings.append("output was cut off by max_tokens (stop_reason=max_tokens)")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return Completion(
            text=text,
            model=data.get("model") if isinstance(data.get("model"), str) else None,
            refused=stop == "refusal",
            finish_reason=stop if isinstance(stop, str) else None,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            warnings=warnings,
        )

    def complete(self, request: CompletionRequest) -> Completion:
        data = http.post_json(
            self.url,
            self.build_payload(request),
            headers=self.headers(),
            timeout=request.timeout,
            policy=self.policy,
            secrets=(self._api_key,),
        )
        return self.parse_response(data)
