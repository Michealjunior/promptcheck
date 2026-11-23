"""``{var}`` substitution for prompts, system prompts and assertion values.

Only ``{identifier}`` is treated as a placeholder, so JSON examples such as
``{"category": "billing"}`` or nested ``}}`` inside a prompt are left alone.
Identifiers follow Python's rules and may use non-ASCII letters (``{名前}``).
Write ``{{name}}`` to produce a literal ``{name}``.

A variable that is missing, or defined without a value (``null``), is an error:
silently substituting an empty string would let assertions such as
``contains: "{invoice_id}"`` pass vacuously.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

IDENTIFIER = re.compile(r"[^\W\d]\w*")
_PLACEHOLDER = re.compile(r"\{\{([^\W\d]\w*)\}\}|\{([^\W\d]\w*)\}")


def is_variable_name(name: object) -> bool:
    """True when ``name`` can be used as ``{name}`` in a template."""
    return isinstance(name, str) and name.isidentifier() and IDENTIFIER.fullmatch(name) is not None


class MissingVariableError(KeyError):
    """Raised when a template references variables that are undefined or null."""

    def __init__(self, names: list[str], nulls: list[str] | None = None) -> None:
        self.names = names
        self.nulls = nulls or []
        super().__init__(", ".join(names + self.nulls))

    def describe(self) -> str:
        """Human-readable reason, e.g. ``{ticket} is not defined in 'vars'``."""
        parts = []
        if self.names:
            listed = ", ".join(f"{{{n}}}" for n in self.names)
            verb = "is" if len(self.names) == 1 else "are"
            parts.append(f"{listed} {verb} not defined in 'vars'")
        if self.nulls:
            listed = ", ".join(f"{{{n}}}" for n in self.nulls)
            verb = "has" if len(self.nulls) == 1 else "have"
            parts.append(f"{listed} {verb} no value (null); give it a value")
        return "; ".join(parts)


def format_value(value: Any) -> str:
    """Turn a YAML scalar (or structure) into the text that is substituted."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def placeholders(template: str) -> list[str]:
    """Return the variable names referenced by ``template`` (escaped ones excluded)."""
    return [m.group(2) for m in _PLACEHOLDER.finditer(template) if m.group(2)]


def render(template: str, variables: Mapping[str, Any]) -> str:
    """Substitute ``{name}`` placeholders, raising if any variable is undefined or null."""
    missing: list[str] = []
    nulls: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        escaped, name = match.group(1), match.group(2)
        if escaped is not None:
            return "{" + escaped + "}"
        if name in variables and variables[name] is not None:
            return format_value(variables[name])
        bucket = nulls if name in variables else missing
        if name not in bucket:
            bucket.append(name)
        return match.group(0)

    result = _PLACEHOLDER.sub(substitute, template)
    if missing or nulls:
        raise MissingVariableError(missing, nulls)
    return result


def render_deep(value: Any, variables: Mapping[str, Any]) -> Any:
    """Render every string inside nested lists and mappings."""
    if isinstance(value, str):
        return render(value, variables)
    if isinstance(value, list):
        return [render_deep(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: render_deep(item, variables) for key, item in value.items()}
    return value
