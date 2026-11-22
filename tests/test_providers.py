from __future__ import annotations

import urllib.request

import pytest

from promptcheck.errors import ConfigError, ProviderError, ProviderHTTPError
from promptcheck.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    ReplayProvider,
    RetryPolicy,
    create_provider,
)
from promptcheck.providers.base import CompletionRequest
from promptcheck.suite import ProviderSettings

from .conftest import EXAMPLE_RECORDINGS, FakeOpener, http_error

KEY = "sk-unit-test-key-0001"
FAST = RetryPolicy(retries=1, backoff=0, jitter=False)


def request(**overrides) -> CompletionRequest:
    values = {
        "case_id": "triage",
        "sample": 0,
        "model": None,
        "system": "Reply in JSON.",
        "messages": ({"role": "user", "content": "Ticket: refund please"},),
        "temperature": None,
        "max_tokens": None,
        "timeout": 9,
    }
    values.update(overrides)
    return CompletionRequest(**values)


OPENAI_OK = {
    "id": "chatcmpl-1",
    "model": "gpt-4.1-mini-2025-04-14",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"category": "billing"}'},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 31, "completion_tokens": 7},
}

ANTHROPIC_OK = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "abc"},
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "there."},
    ],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 5},
}


@pytest.fixture
def opener(monkeypatch):
    fake = FakeOpener()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


# ----------------------------------------------------------------- OpenAI-compatible


def test_openai_request_building_and_parsing(opener):
    opener.script.append(OPENAI_OK)
    provider = OpenAICompatibleProvider(model="gpt-4.1-mini", environ={"OPENAI_API_KEY": KEY}, policy=FAST)
    completion = provider.complete(request(temperature=0.0, max_tokens=300))
    sent = opener.requests[0]
    assert sent.full_url == "https://api.openai.com/v1/chat/completions"
    assert opener.header("Authorization") == f"Bearer {KEY}"
    assert opener.timeouts == [9]
    assert opener.body() == {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "Reply in JSON."},
            {"role": "user", "content": "Ticket: refund please"},
        ],
        "temperature": 0.0,
        "max_completion_tokens": 300,
    }
    assert completion.text == '{"category": "billing"}'
    assert completion.model == "gpt-4.1-mini-2025-04-14"
    assert (completion.input_tokens, completion.output_tokens) == (31, 7)
    assert not completion.refused
    assert completion.warnings == []


def test_openai_omits_unset_sampling_params_and_system(opener):
    opener.script.append(OPENAI_OK)
    provider = OpenAICompatibleProvider(environ={"OPENAI_API_KEY": KEY})
    provider.complete(request(model="case-model", system=None))
    assert opener.body() == {
        "model": "case-model",
        "messages": [{"role": "user", "content": "Ticket: refund please"}],
    }


def test_compatible_servers_use_max_tokens_and_custom_base_url(opener):
    opener.script.append(OPENAI_OK)
    provider = OpenAICompatibleProvider(
        "openai", model="llama3.2", base_url="http://localhost:8000/v1/", environ={}
    )
    provider.complete(request(max_tokens=50))
    assert opener.requests[0].full_url == "http://localhost:8000/v1/chat/completions"
    assert opener.body()["max_tokens"] == 50
    assert opener.header("Authorization") is None
    assert provider.label == "openai (http://localhost:8000/v1)"


@pytest.mark.parametrize(
    ("preset", "url", "env"),
    [
        ("openrouter", "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
        ("groq", "https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    ],
)
def test_hosted_presets(opener, preset, url, env):
    opener.script.append(OPENAI_OK)
    provider = OpenAICompatibleProvider(preset, model="m", environ={env: KEY})
    provider.complete(request())
    assert opener.requests[0].full_url == url
    assert opener.header("Authorization") == f"Bearer {KEY}"
    assert provider.label == preset


def test_ollama_needs_no_key(opener):
    opener.script.append(OPENAI_OK)
    provider = OpenAICompatibleProvider("ollama", model="llama3.2", environ={})
    provider.complete(request())
    assert opener.requests[0].full_url == "http://localhost:11434/v1/chat/completions"
    assert opener.header("Authorization") is None


def test_missing_key_is_a_config_error_that_names_the_variable():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY is not set"):
        OpenAICompatibleProvider(model="m", environ={})
    with pytest.raises(ConfigError, match="MY_GATEWAY_KEY is not set"):
        OpenAICompatibleProvider(base_url="http://gw.local/v1", api_key_env="MY_GATEWAY_KEY", environ={})


def test_openai_refusal_and_truncation():
    provider = OpenAICompatibleProvider(environ={"OPENAI_API_KEY": KEY})
    refused = provider.parse_response(
        {
            "choices": [
                {"message": {"content": None, "refusal": "I can't help with that."}, "finish_reason": "stop"}
            ]
        }
    )
    assert refused.refused and refused.text == "I can't help with that."
    cut = provider.parse_response(
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
    )
    assert cut.warnings == ["output was cut off by max_tokens (finish_reason=length)"]
    parts = provider.parse_response(
        {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}
            ]
        }
    )
    assert parts.text == "ab"


def test_openai_bad_shape_raises():
    provider = OpenAICompatibleProvider(environ={"OPENAI_API_KEY": KEY})
    with pytest.raises(ProviderError, match="no 'choices'"):
        provider.parse_response({"object": "error"})


def test_openai_missing_model():
    provider = OpenAICompatibleProvider(environ={"OPENAI_API_KEY": KEY})
    with pytest.raises(ProviderError, match="no model configured"):
        provider.build_payload(request())


def test_api_key_never_appears_in_errors(opener):
    opener.script.append(http_error(401, {"error": {"message": f"Incorrect API key provided: {KEY}."}}))
    provider = OpenAICompatibleProvider(model="m", environ={"OPENAI_API_KEY": KEY}, policy=FAST)
    with pytest.raises(ProviderHTTPError) as info:
        provider.complete(request())
    assert KEY not in str(info.value)
    assert "Incorrect API key provided: ***." in str(info.value)


def test_openai_retries_rate_limits(opener):
    opener.script.extend([http_error(429), OPENAI_OK])
    provider = OpenAICompatibleProvider(model="m", environ={"OPENAI_API_KEY": KEY}, policy=FAST)
    assert provider.complete(request()).text == '{"category": "billing"}'
    assert len(opener.requests) == 2


# ----------------------------------------------------------------- Anthropic


def test_anthropic_request_building_and_parsing(opener):
    opener.script.append(ANTHROPIC_OK)
    provider = AnthropicProvider(model="claude-opus-5-5", environ={"ANTHROPIC_API_KEY": KEY})
    completion = provider.complete(request(max_tokens=800))
    assert opener.requests[0].full_url == "https://api.anthropic.com/v1/messages"
    assert opener.header("x-api-key") == KEY
    assert opener.header("anthropic-version") == "2026-10-21"
    assert opener.body() == {
        "model": "claude-opus-5-5",
        "max_tokens": 800,
        "system": "Reply in JSON.",
        "messages": [{"role": "user", "content": "Ticket: refund please"}],
    }
    assert completion.text == "Hello there."
    assert completion.model == "claude-opus-5-5"
    assert (completion.input_tokens, completion.output_tokens) == (12, 5)
    assert not completion.refused


def test_anthropic_defaults_max_tokens_and_drops_temperature_on_current_api(opener):
    opener.script.append(ANTHROPIC_OK)
    provider = AnthropicProvider(model="claude-opus-5-5", environ={"ANTHROPIC_API_KEY": KEY})
    provider.complete(request(temperature=0.0, system=None))
    body = opener.body()
    assert body["max_tokens"] == 4096
    assert "temperature" not in body
    assert "system" not in body
    assert len(provider.notices) == 1
    assert "temperature is not sent" in provider.notices[0]


def test_anthropic_legacy_version_sends_temperature(opener):
    opener.script.append(ANTHROPIC_OK)
    provider = AnthropicProvider(
        model="claude-haiku-4-5", api_version="2023-06-01", environ={"ANTHROPIC_API_KEY": KEY}
    )
    provider.complete(request(temperature=0.2))
    assert opener.body()["temperature"] == 0.2
    assert opener.header("anthropic-version") == "2023-06-01"
    assert provider.notices == []


def test_anthropic_refusal_and_max_tokens_stop_reasons():
    provider = AnthropicProvider(environ={"ANTHROPIC_API_KEY": KEY})
    refused = provider.parse_response({"content": [], "stop_reason": "refusal"})
    assert refused.refused and refused.text == ""
    cut = provider.parse_response({"content": [{"type": "text", "text": "par"}], "stop_reason": "max_tokens"})
    assert cut.warnings == ["output was cut off by max_tokens (stop_reason=max_tokens)"]


def test_anthropic_custom_key_env_and_base_url(opener):
    opener.script.append(ANTHROPIC_OK)
    provider = AnthropicProvider(
        model="m",
        base_url="https://gateway.test/anthropic/",
        api_key_env="GATEWAY_KEY",
        environ={"GATEWAY_KEY": KEY},
    )
    provider.complete(request())
    assert opener.requests[0].full_url == "https://gateway.test/anthropic/v1/messages"


def test_anthropic_missing_key():
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicProvider(model="m", environ={})


def test_anthropic_errors_are_redacted(opener):
    opener.script.append(
        http_error(400, {"type": "error", "error": {"type": "invalid_request_error", "message": KEY}})
    )
    provider = AnthropicProvider(model="m", environ={"ANTHROPIC_API_KEY": KEY})
    with pytest.raises(ProviderHTTPError) as info:
        provider.complete(request())
    assert KEY not in str(info.value)


def test_anthropic_bad_shape_raises():
    provider = AnthropicProvider(environ={"ANTHROPIC_API_KEY": KEY})
    with pytest.raises(ProviderError, match="no 'content'"):
        provider.parse_response({"type": "error"})


# ----------------------------------------------------------------- factory


def test_factory_picks_providers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    settings = ProviderSettings(name="openai", model="gpt-4.1-mini", base_url="http://suite.local/v1")
    openai = create_provider(settings)
    assert isinstance(openai, OpenAICompatibleProvider)
    assert openai.base_url == "http://suite.local/v1"
    anthropic = create_provider(settings, name="anthropic", model="claude-opus-5-5")
    assert isinstance(anthropic, AnthropicProvider)
    assert anthropic.base_url == "https://api.anthropic.com", "suite base_url must not leak across providers"
    assert anthropic.model == "claude-opus-5-5"
    replay = create_provider(settings, replay=EXAMPLE_RECORDINGS)
    assert isinstance(replay, ReplayProvider)


def test_factory_replay_misuse():
    with pytest.raises(ConfigError, match="needs --replay"):
        create_provider(ProviderSettings(name="replay"))
    with pytest.raises(ConfigError, match="only works with --provider replay"):
        create_provider(ProviderSettings(), name="openai", replay=EXAMPLE_RECORDINGS)
