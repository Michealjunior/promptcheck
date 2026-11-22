"""OpenAI-compatible Chat Completions provider.

Works with OpenAI itself and with any server that speaks the same API, such as
OpenRouter, Groq, Ollama, vLLM or LM Studio, by changing ``base_url``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from promptcheck.errors import ConfigError, ProviderError
from promptcheck.providers import http
from promptcheck.providers.base import Completion, CompletionRequest, Provider


@dataclass(frozen=True)
class Preset:
    base_url: str
    api_key_env: str | None
    key_required: bool


PRESETS: dict[str, Preset] = {
    "openai": Preset("https://api.openai.com/v1", "OPENAI_API_KEY", True),
    "openrouter": Preset("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", True),
    "groq": Preset("https://api.groq.com/openai/v1", "GROQ_API_KEY", True),
    "ollama": Preset("http://localhost:11434/v1", None, False),
}


class OpenAICompatibleProvider(Provider):
    """``POST {base_url}/chat/completions``."""

    def __init__(
        self,
        preset: str = "openai",
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        policy: http.RetryPolicy | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        if preset not in PRESETS:
            raise ConfigError(f"unknown OpenAI-compatible preset {preset!r}")
        defaults = PRESETS[preset]
        env = os.environ if environ is None else environ
        self.name = preset
        self.model = model
        self.base_url = (base_url or defaults.base_url).rstrip("/")
        self.policy = policy or http.RetryPolicy()
        self.api_key_env = api_key_env or defaults.api_key_env
        self._api_key = env.get(self.api_key_env) if self.api_key_env else None
        uses_default_endpoint = self.base_url == defaults.base_url.rstrip("/")
        key_required = bool(api_key_env) or (defaults.key_required and uses_default_endpoint)
        if key_required and not self._api_key:
            raise ConfigError(
                f"{self.api_key_env} is not set. Export it (export {self.api_key_env}=...) "
                "or run offline with --provider replay --replay recordings.json"
            )
        host = urlparse(self.base_url).hostname or ""
        # OpenAI's own API has moved to max_completion_tokens; compatible servers use max_tokens.
        self.max_tokens_field = "max_completion_tokens" if host == "api.openai.com" else "max_tokens"

    @property
    def label(self) -> str:
        if self.base_url != PRESETS[self.name].base_url:
            return f"{self.name} ({self.base_url})"
        return self.name

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self.name == "openrouter":
            headers["X-Title"] = "promptcheck"
        return headers

    def build_payload(self, request: CompletionRequest) -> dict[str, Any]:
        model = request.model or self.model
        if not model:
            raise ProviderError("no model configured; set defaults.model in the suite or pass --model")
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(dict(m) for m in request.messages)
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload[self.max_tokens_field] = request.max_tokens
        return payload

    def parse_response(self, data: dict[str, Any]) -> Completion:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError("unexpected response: no 'choices' in the completion")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # some servers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        text = content if isinstance(content, str) else ""
        refusal = message.get("refusal")
        finish = choice.get("finish_reason")
        refused = bool(refusal) or finish == "content_filter"
        if not text and isinstance(refusal, str):
            text = refusal
        warnings = []
        if finish == "length":
            warnings.append("output was cut off by max_tokens (finish_reason=length)")
        if finish == "content_filter":
            warnings.append("the provider's content filter stopped this output")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return Completion(
            text=text,
            model=data.get("model") if isinstance(data.get("model"), str) else None,
            refused=refused,
            finish_reason=finish if isinstance(finish, str) else None,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
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
