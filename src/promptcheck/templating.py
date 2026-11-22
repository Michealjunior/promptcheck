"""``{var}`` substitution for prompts, system prompts and assertion values.

Only ``{identifier}`` is treated as a placeholder, so JSON examples such as
``{"category": "billing"}`` or nested ``}}`` inside a prompt are left alone.
Write ``{{name}}`` to produce a literal ``{name}``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_PLACEHOLDER = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}")


class MissingVariableError(KeyError):
    """Raised when a template references variables that are not defined."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__(", ".join(names))


def format_value(value: Any) -> str:
    """Turn a YAML scalar (or structure) into the text that is substituted."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def placeholders(template: str) -> list[str]:
    """Return the variable names referenced by ``template`` (escaped ones excluded)."""
    return [m.group(2) for m in _PLACEHOLDER.finditer(template) if m.group(2)]


def render(template: str, variables: Mapping[str, Any]) -> str:
    """Substitute ``{name}`` placeholders, raising if any variable is undefined."""
    missing: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        escaped, name = match.group(1), match.group(2)
        if escaped is not None:
            return "{" + escaped + "}"
        if name in variables:
            return format_value(variables[name])
        if name not in missing:
            missing.append(name)
        return match.group(0)

    result = _PLACEHOLDER.sub(substitute, template)
    if missing:
        raise MissingVariableError(missing)
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
