"""Loading and validating YAML test suites."""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from promptcheck.assertions import (
    NO_TEMPLATE_KINDS,
    AssertionConfigError,
    Check,
    build_check,
    split_entry,
)
from promptcheck.errors import SuiteError
from promptcheck.templating import MissingVariableError, render, render_deep

PROVIDERS = ("openai", "openrouter", "groq", "ollama", "anthropic", "replay")

TOP_LEVEL_KEYS = ("name", "description", "defaults", "cases")
DEFAULT_KEYS = (
    "provider",
    "model",
    "base_url",
    "api_key_env",
    "anthropic_version",
    "temperature",
    "max_tokens",
    "system",
    "repeat",
    "pass_rate",
    "timeout",
    "vars",
)
CASE_KEYS = (
    "id",
    "description",
    "prompt",
    "messages",
    "system",
    "vars",
    "assert",
    "repeat",
    "pass_rate",
    "model",
    "temperature",
    "max_tokens",
)
MESSAGE_ROLES = ("user", "assistant")
MAX_REPEAT = 100


@dataclass(frozen=True)
class ProviderSettings:
    """Provider configuration from the suite's ``defaults`` block."""

    name: str = "openai"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    anthropic_version: str | None = None


@dataclass
class Case:
    """One fully rendered test case."""

    id: str
    messages: list[dict[str, str]]
    system: str | None
    checks: list[Check]
    repeat: int = 1
    pass_rate: float = 1.0
    description: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    variables: dict[str, Any] = field(default_factory=dict)


@dataclass
class Suite:
    """A parsed suite file."""

    name: str
    path: Path
    cases: list[Case]
    provider: ProviderSettings
    timeout: float = 60.0
    description: str | None = None

    @property
    def sample_count(self) -> int:
        return sum(case.repeat for case in self.cases)


def request_fingerprint(model: str | None, system: str | None, messages: list[dict[str, str]]) -> str:
    """Stable short hash of what was sent, used to detect stale recordings."""
    canonical = json.dumps(
        {"model": model, "system": system, "messages": messages},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- helpers


class _Where:
    """Builds error messages that point at the file and case."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def error(self, scope: str, message: str) -> SuiteError:
        return SuiteError(f"{self.path}: {scope}: {message}")


def _suggest(name: str, options: tuple[str, ...]) -> str:
    close = difflib.get_close_matches(name, list(options), n=1, cutoff=0.6)
    return f" (did you mean '{close[0]}'?)" if close else ""


def _check_keys(where: _Where, scope: str, data: dict[str, Any], allowed: tuple[str, ...]) -> None:
    for key in data:
        if key not in allowed:
            hint = _suggest(str(key), allowed)
            if key == "assertions":
                hint = " (did you mean 'assert'?)"
            raise where.error(scope, f"unknown key '{key}'{hint}; allowed keys: {', '.join(allowed)}")


def _type_name(value: Any) -> str:
    if value is None:
        return "nothing"
    return {
        bool: "a boolean",
        int: "a number",
        float: "a number",
        str: "a string",
        list: "a list",
        dict: "a mapping",
    }.get(type(value), type(value).__name__)


def _optional_str(where: _Where, scope: str, key: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise where.error(scope, f"'{key}' must be a non-empty string, got {_type_name(value)}")
    return value


def _optional_number(
    where: _Where, scope: str, key: str, value: Any, low: float, high: float
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise where.error(scope, f"'{key}' must be a number between {low:g} and {high:g}, got {value!r}")
    return float(value)


def _optional_positive_int(
    where: _Where, scope: str, key: str, value: Any, high: int | None = None
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise where.error(scope, f"'{key}' must be a positive whole number, got {value!r}")
    if high is not None and value > high:
        raise where.error(scope, f"'{key}' must be at most {high}, got {value}")
    return value


def parse_rate(value: Any) -> float:
    """Parse a pass rate given as ``0.8``, ``80%`` or ``"0.8"``. Raises ValueError."""
    if isinstance(value, bool):
        raise ValueError(value)
    if isinstance(value, str):
        text = value.strip()
        rate = float(text[:-1]) / 100 if text.endswith("%") else float(text)
    elif isinstance(value, (int, float)):
        rate = float(value)
    else:
        raise ValueError(value)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(value)
    return rate


def _optional_rate(where: _Where, scope: str, key: str, value: Any) -> float | None:
    if value is None:
        return None
    try:
        return parse_rate(value)
    except (TypeError, ValueError):
        raise where.error(
            scope, f"'{key}' must be a fraction between 0 and 1 (or a percentage like 80%), got {value!r}"
        ) from None


def _vars(where: _Where, scope: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise where.error(scope, f"'vars' must be a mapping of names to values, got {_type_name(value)}")
    for name in value:
        if not isinstance(name, str) or not name.isidentifier():
            raise where.error(scope, f"variable name {name!r} must be a valid identifier")
    return dict(value)


def _render(where: _Where, scope: str, what: str, template: str, variables: dict[str, Any]) -> str:
    try:
        return render(template, variables)
    except MissingVariableError as exc:
        names = ", ".join(f"{{{n}}}" for n in exc.names)
        plural = "variables" if len(exc.names) > 1 else "variable"
        raise where.error(
            scope,
            f"{what} uses {names} but no {plural} named {', '.join(exc.names)} "
            "is defined in 'vars' (write {{name}} for a literal brace)",
        ) from None


# --------------------------------------------------------------------------- loading


def load_suite(path: str | Path) -> Suite:
    """Read, validate and render a suite file."""
    suite_path = Path(path)
    try:
        text = suite_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SuiteError(f"{suite_path}: file not found") from None
    except OSError as exc:
        raise SuiteError(f"{suite_path}: cannot read file ({exc.strerror})") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        problem = getattr(exc, "problem", None) or "invalid YAML"
        raise SuiteError(f"{suite_path}: YAML syntax error{location}: {problem}") from None
    return parse_suite(data, suite_path)


def parse_suite(data: Any, path: str | Path = "<suite>") -> Suite:
    """Validate an already-parsed YAML document."""
    suite_path = Path(path)
    where = _Where(suite_path)
    if not isinstance(data, dict):
        raise where.error("suite", "the file must contain a mapping with 'name' and 'cases'")
    _check_keys(where, "suite", data, TOP_LEVEL_KEYS)

    name = data.get("name")
    if name is None:
        name = suite_path.stem
    elif not isinstance(name, str) or not name.strip():
        raise where.error("suite", "'name' must be a non-empty string")
    description = _optional_str(where, "suite", "description", data.get("description"))

    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise where.error("defaults", f"must be a mapping, got {_type_name(defaults)}")
    _check_keys(where, "defaults", defaults, DEFAULT_KEYS)

    provider_name = defaults.get("provider", "openai")
    if provider_name not in PROVIDERS:
        raise where.error(
            "defaults",
            f"unknown provider {provider_name!r}{_suggest(str(provider_name), PROVIDERS)}; "
            f"choose one of {', '.join(PROVIDERS)}",
        )
    provider = ProviderSettings(
        name=provider_name,
        model=_optional_str(where, "defaults", "model", defaults.get("model")),
        base_url=_optional_str(where, "defaults", "base_url", defaults.get("base_url")),
        api_key_env=_optional_str(where, "defaults", "api_key_env", defaults.get("api_key_env")),
        anthropic_version=_optional_str(
            where, "defaults", "anthropic_version", defaults.get("anthropic_version")
        ),
    )
    default_system = _optional_str(where, "defaults", "system", defaults.get("system"))
    default_temperature = _optional_number(
        where, "defaults", "temperature", defaults.get("temperature"), 0, 2
    )
    default_max_tokens = _optional_positive_int(where, "defaults", "max_tokens", defaults.get("max_tokens"))
    default_repeat = _optional_positive_int(where, "defaults", "repeat", defaults.get("repeat"), MAX_REPEAT)
    default_rate = _optional_rate(where, "defaults", "pass_rate", defaults.get("pass_rate"))
    timeout = _optional_number(where, "defaults", "timeout", defaults.get("timeout"), 1, 3600)
    default_vars = _vars(where, "defaults", defaults.get("vars"))

    raw_cases = data.get("cases")
    if raw_cases is None:
        raise where.error("suite", "missing 'cases'; add a list of test cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise where.error("suite", "'cases' must be a non-empty list")

    cases: list[Case] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_cases, start=1):
        case = _parse_case(
            where,
            position,
            raw,
            default_system=default_system,
            default_vars=default_vars,
            default_repeat=default_repeat or 1,
            default_rate=1.0 if default_rate is None else default_rate,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
        )
        if case.id in seen:
            raise where.error(f"case '{case.id}'", "duplicate case id; every case needs a unique id")
        seen.add(case.id)
        cases.append(case)

    return Suite(
        name=name.strip(),
        path=suite_path,
        cases=cases,
        provider=provider,
        timeout=60.0 if timeout is None else timeout,
        description=description,
    )


def _parse_case(
    where: _Where,
    position: int,
    raw: Any,
    *,
    default_system: str | None,
    default_vars: dict[str, Any],
    default_repeat: int,
    default_rate: float,
    default_temperature: float | None,
    default_max_tokens: int | None,
) -> Case:
    anonymous = f"case #{position}"
    if not isinstance(raw, dict):
        raise where.error(anonymous, f"each case must be a mapping, got {_type_name(raw)}")
    case_id = raw.get("id")
    if case_id is None:
        raise where.error(anonymous, "missing 'id'; give every case a short unique id")
    if isinstance(case_id, bool) or not isinstance(case_id, (str, int)) or not str(case_id).strip():
        raise where.error(anonymous, "'id' must be a non-empty string")
    case_id = str(case_id).strip()
    scope = f"case '{case_id}'"
    _check_keys(where, scope, raw, CASE_KEYS)

    variables = {**default_vars, **_vars(where, scope, raw.get("vars"))}

    has_prompt = "prompt" in raw and raw["prompt"] is not None
    has_messages = "messages" in raw and raw["messages"] is not None
    if has_prompt and has_messages:
        raise where.error(scope, "use either 'prompt' or 'messages', not both")
    if not has_prompt and not has_messages:
        raise where.error(scope, "missing 'prompt' (or 'messages'); nothing to send to the model")

    messages: list[dict[str, str]]
    if has_prompt:
        prompt = raw["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise where.error(scope, "'prompt' must be a non-empty string")
        messages = [{"role": "user", "content": _render(where, scope, "prompt", prompt, variables)}]
    else:
        messages = _parse_messages(where, scope, raw["messages"], variables)

    system_raw = raw.get("system", default_system)
    system = None
    if system_raw is not None:
        if not isinstance(system_raw, str):
            raise where.error(scope, f"'system' must be a string, got {_type_name(system_raw)}")
        system = _render(where, scope, "system prompt", system_raw, variables) or None

    raw_asserts = raw.get("assert")
    if raw_asserts is None:
        raise where.error(scope, "missing 'assert'; add at least one assertion such as '- json_valid'")
    if not isinstance(raw_asserts, list) or not raw_asserts:
        raise where.error(scope, "'assert' must be a non-empty list of assertions")
    checks: list[Check] = []
    for number, entry in enumerate(raw_asserts, start=1):
        try:
            kind, arg = split_entry(entry)
            if kind not in NO_TEMPLATE_KINDS:
                arg = render_deep(arg, variables)
            checks.append(build_check(kind, arg))
        except AssertionConfigError as exc:
            raise where.error(scope, f"assertion #{number}: {exc}") from None
        except MissingVariableError as exc:
            names = ", ".join(f"{{{n}}}" for n in exc.names)
            raise where.error(
                scope, f"assertion #{number} uses {names}, which is not defined in 'vars'"
            ) from None

    repeat = _optional_positive_int(where, scope, "repeat", raw.get("repeat"), MAX_REPEAT)
    rate = _optional_rate(where, scope, "pass_rate", raw.get("pass_rate"))
    temperature = _optional_number(where, scope, "temperature", raw.get("temperature"), 0, 2)
    max_tokens = _optional_positive_int(where, scope, "max_tokens", raw.get("max_tokens"))

    return Case(
        id=case_id,
        description=_optional_str(where, scope, "description", raw.get("description")),
        messages=messages,
        system=system,
        checks=checks,
        repeat=repeat or default_repeat,
        pass_rate=default_rate if rate is None else rate,
        model=_optional_str(where, scope, "model", raw.get("model")),
        temperature=default_temperature if temperature is None else temperature,
        max_tokens=default_max_tokens if max_tokens is None else max_tokens,
        variables=variables,
    )


def _parse_messages(where: _Where, scope: str, raw: Any, variables: dict[str, Any]) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise where.error(scope, "'messages' must be a non-empty list of {role, content} mappings")
    messages = []
    for number, message in enumerate(raw, start=1):
        label = f"message #{number}"
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise where.error(scope, f"{label} must have exactly the keys 'role' and 'content'")
        role, content = message["role"], message["content"]
        if role == "system":
            raise where.error(scope, f"{label}: put system prompts in 'system', not in 'messages'")
        if role not in MESSAGE_ROLES:
            raise where.error(scope, f"{label}: role must be 'user' or 'assistant', got {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise where.error(scope, f"{label}: 'content' must be a non-empty string")
        messages.append({"role": role, "content": _render(where, scope, label, content, variables)})
    if messages[-1]["role"] != "user":
        raise where.error(scope, "the last message must have role 'user'")
    return messages
