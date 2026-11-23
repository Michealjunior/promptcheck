"""Assertion types that can be attached to a case.

Every assertion is written in YAML as a single-key mapping, for example
``- max_words: 60`` or ``- json_field: {path: priority, equals: urgent}``.
Assertions that take no argument may also be written as a bare string
(``- json_valid``).
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any, ClassVar

REFUSAL_WINDOW = 300
"""Number of leading characters scanned by the refusal heuristic."""

_REFUSAL_VERBS = (
    "help|assist|provide|share|do|comply|fulfil|fulfill|reveal|disclose|give|answer|support|"
    "create|generate|write|continue|engage|discuss|tell|process|complete|offer|show|make|"
    "produce|print|repeat|follow|accommodate|participate|release|expose|output|grant|perform"
)
_SOFTENER = (
    r"(?:(?:i'm|i am|we're|we are)\s+(?:so\s+|very\s+|really\s+|truly\s+)?(?:sorry|afraid)"
    r"|(?:i|we)\s+(?:sincerely\s+)?apologi[sz]e|(?:so\s+|very\s+|really\s+)?sorry"
    r"|unfortunately|regrettably|apologies)"
)
_INABILITY = (
    r"(?:can't|cannot|can\s+not|won't|will\s+not|must\s+not|am\s+(?:not\s+able|unable|not\s+going"
    r"|not\s+allowed|not\s+permitted)\s+to)"
)
_REFUSAL_START = re.compile(
    rf"(?:{_SOFTENER}\s*[,.!:;\u2014-]?\s*(?:but\s+|however,?\s+)?)?"
    r"(?:"
    rf"i\s+{_INABILITY}\s+(?:be\s+able\s+to\s+)?(?:{_REFUSAL_VERBS})\b"
    rf"|i'm\s+(?:not\s+able|unable|not\s+going|not\s+allowed|not\s+permitted)\s+to\s+(?:{_REFUSAL_VERBS})\b"
    r"|i\s+(?:must|have\s+to|need\s+to|will|'ll)\s+(?:respectfully\s+|politely\s+)?decline\b"
    r"|i'll\s+(?:have\s+to\s+)?(?:respectfully\s+)?decline\b"
    r"|i\s+(?:respectfully\s+)?decline\s+to\b"
    r"|(?:i'm|i\s+am)\s+not\s+comfortable\b"
    r"|i\s+don't\s+feel\s+comfortable\b"
    rf"|(?:that's|that\s+is|this\s+is)\s+not\s+something\s+i\s+(?:can|am\s+able\s+to|will)\s+"
    rf"(?:{_REFUSAL_VERBS})\b"
    r")"
)
"""A first-person refusal at the start of a sentence, optionally after an apology."""

_REDIRECT = re.compile(
    r"\b(?:but|however,?|instead,?)\s+(?:you\s+can|you\s+could|here's|here\s+is|here\s+are)\b"
)
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u00b4": "'", "`": "'"})
_LEADING_MARKUP = re.compile(r"^[\s*_#>]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n\s*\n")

JSON_TYPES: dict[str, str] = {
    "string": "a string",
    "number": "a number",
    "integer": "an integer",
    "boolean": "a boolean",
    "array": "an array",
    "object": "an object",
    "null": "null",
}


class AssertionConfigError(ValueError):
    """An assertion entry in a suite is malformed."""


@dataclass(frozen=True)
class AssertionResult:
    """Outcome of one assertion against one output."""

    kind: str
    label: str
    passed: bool
    detail: str = ""


def quote(text: str, limit: int = 60) -> str:
    """Quote ``text`` for messages, shortening it to ``limit`` characters."""
    flat = " ".join(str(text).split()) if "\n" in str(text) else str(text)
    if len(flat) > limit:
        flat = flat[: limit - 1].rstrip() + "…"
    return json.dumps(flat, ensure_ascii=False)


def _normalise(text: str) -> str:
    text = text[:REFUSAL_WINDOW].translate(_APOSTROPHES).lower()
    return _LEADING_MARKUP.sub("", text)


def detect_refusal(output: str) -> str | None:
    """Return the refusal wording that opens ``output``, if any.

    English-only heuristic. A reply counts as a refusal when its first sentence,
    or its second sentence after a short opener such as "Thanks for asking!",
    starts with a first-person refusal: optionally an apology ("I'm sorry, but",
    "Unfortunately,") followed by "I can't / cannot / won't / am unable to /
    am not able to / am not going to" plus a verb such as help, assist, provide,
    share, do, comply, reveal or disclose, or by "I must decline". Refusal
    wording that appears later, inside quotes, or that is followed by a
    redirection ("..., but you can ...", "..., but here's how") does not count.
    Only the first :data:`REFUSAL_WINDOW` characters are examined.
    """
    text = _normalise(output)
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()][:2]
    for index, sentence in enumerate(sentences):
        sentence = " ".join(sentence.split())
        if index == 1:
            opener = " ".join(sentences[0].split())
            if len(opener) > 80 or ":" in opener:
                break
        match = _REFUSAL_START.match(sentence)
        if match:
            if _REDIRECT.search(sentence, match.end()):
                return None
            return match.group(0)
    return None


def word_count(output: str) -> int:
    return len(output.split())


class _NonStandardNumber(ValueError):
    pass


def _reject_constant(name: str) -> Any:
    # Python's json module accepts NaN, Infinity and -Infinity; strict JSON does not.
    raise _NonStandardNumber(name)


def parse_json_output(output: str) -> tuple[Any, str | None]:
    """Parse ``output`` as JSON. Returns ``(value, error_message)``."""
    text = output.strip()
    try:
        return json.loads(text, parse_constant=_reject_constant), None
    except _NonStandardNumber as exc:
        return None, f"output is not valid JSON: {exc} is not allowed (JSON has no NaN or Infinity)"
    except json.JSONDecodeError as exc:
        if not text:
            message = "output is empty, expected JSON"
        elif text.startswith("```"):
            message = "output is wrapped in a Markdown code fence, so it is not valid JSON"
        else:
            message = f"output is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}"
        return None, message


def _scalar_text(value: Any, kind: str, what: str = "a string", allow_empty: bool = True) -> str:
    if isinstance(value, bool) or value is None or not isinstance(value, (str, int, float)):
        raise AssertionConfigError(f"'{kind}' expects {what}, got {_describe(value)}")
    text = str(value)
    if not allow_empty and text == "":
        raise AssertionConfigError(
            f"'{kind}' needs a non-empty string; an empty string would match every output"
        )
    return text


def _text_list(value: Any, kind: str) -> list[str]:
    if isinstance(value, list):
        if not value:
            raise AssertionConfigError(f"'{kind}' expects a string or a non-empty list of strings")
        return [_scalar_text(item, kind, "strings", allow_empty=False) for item in value]
    return [_scalar_text(value, kind, "a string or a list of strings", allow_empty=False)]


def _positive_int(value: Any, kind: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AssertionConfigError(f"'{kind}' expects a positive integer, got {_describe(value)}")
    return value


def _flag(value: Any, kind: str) -> None:
    if value not in (True, None):
        raise AssertionConfigError(f"'{kind}' takes no argument; write '- {kind}' or '- {kind}: true'")


def _describe(value: Any) -> str:
    if value is None:
        return "nothing"
    if isinstance(value, bool):
        return f"the boolean {str(value).lower()}"
    if isinstance(value, str):
        return f"the string {quote(value, 40)}"
    if isinstance(value, (int, float)):
        return f"the number {value}"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "a mapping"
    return type(value).__name__


def _context(output: str, needle_at: int, needle_len: int, radius: int = 24) -> str:
    start = max(0, needle_at - radius)
    end = min(len(output), needle_at + needle_len + radius)
    snippet = " ".join(output[start:end].split())
    return ("…" if start else "") + snippet + ("…" if end < len(output) else "")


class Check:
    """Base class for an assertion type."""

    kind: ClassVar[str]
    takes_argument: ClassVar[bool] = True

    def __init__(self, arg: Any) -> None:
        self.arg = arg

    def label(self) -> str:
        raise NotImplementedError

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        raise NotImplementedError

    def run(self, output: str, refused: bool = False) -> AssertionResult:
        passed, detail = self.evaluate(output, refused)
        return AssertionResult(self.kind, self.label(), passed, detail)


class Contains(Check):
    kind = "contains"

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        self.needles = _text_list(arg, self.kind)

    def label(self) -> str:
        return f"{self.kind} " + ", ".join(quote(n, 40) for n in self.needles)

    def _find(self, output: str, needle: str) -> bool:
        return needle in output

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        missing = [n for n in self.needles if not self._find(output, n)]
        if missing:
            return False, "missing " + ", ".join(quote(n, 40) for n in missing)
        return True, ""


class IContains(Contains):
    kind = "icontains"

    def _find(self, output: str, needle: str) -> bool:
        return needle.casefold() in output.casefold()


class NotContains(Check):
    kind = "not_contains"

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        self.needles = _text_list(arg, self.kind)

    def label(self) -> str:
        return f"{self.kind} " + ", ".join(quote(n, 40) for n in self.needles)

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        for needle in self.needles:
            at = output.find(needle)
            if at >= 0:
                return False, f"found {quote(needle, 40)} in {quote(_context(output, at, len(needle)), 90)}"
        return True, ""


_VACUITY_PROBES = ("", "x", "\n", "Lorem ipsum dolor sit amet", "{}", "0")


class Regex(Check):
    kind = "regex"

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        pattern = _scalar_text(arg, self.kind, "a regular expression string", allow_empty=False)
        try:
            self.pattern = re.compile(pattern)
        except re.error as exc:
            raise AssertionConfigError(f"'regex' pattern {quote(pattern)} is invalid: {exc}") from exc
        if all(self.pattern.search(probe) for probe in _VACUITY_PROBES):
            raise AssertionConfigError(
                f"'regex' pattern {quote(pattern)} matches any text (even an empty output), "
                "so it can never fail"
            )

    def label(self) -> str:
        return f"regex /{self.pattern.pattern}/"

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        if self.pattern.search(output):
            return True, ""
        return False, f"no match for /{self.pattern.pattern}/"


class Equals(Check):
    kind = "equals"

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        self.expected = _scalar_text(arg, self.kind)

    def label(self) -> str:
        return f"equals {quote(self.expected, 40)}"

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        if output.strip() == self.expected.strip():
            return True, ""
        return False, f"got {quote(output.strip())}"


class StartsWith(Check):
    kind = "starts_with"

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        self.prefix = _scalar_text(arg, self.kind, allow_empty=False)

    def label(self) -> str:
        return f"starts_with {quote(self.prefix, 40)}"

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        if output.lstrip().startswith(self.prefix):
            return True, ""
        return False, f"output starts with {quote(output.lstrip()[: max(len(self.prefix), 20)], 40)}"


class OneOf(Check):
    kind = "one_of"

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        if not isinstance(arg, list) or not arg:
            raise AssertionConfigError("'one_of' expects a non-empty list of allowed outputs")
        self.options = [_scalar_text(item, self.kind, "a list of strings") for item in arg]

    def label(self) -> str:
        return "one_of " + ", ".join(quote(o, 30) for o in self.options)

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        if output.strip() in [o.strip() for o in self.options]:
            return True, ""
        return False, f"got {quote(output.strip())}"


class JsonValid(Check):
    kind = "json_valid"
    takes_argument = False

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        _flag(arg, self.kind)

    def label(self) -> str:
        return "json_valid"

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        _, error = parse_json_output(output)
        return (error is None), (error or "")


_MISSING = object()
_PATH_INDEX = re.compile(r"\[(\d+)\]")


def split_path(path: str) -> list[str]:
    """Split ``a.b[0].c`` (or ``a.b.0.c``) into segments."""
    normalised = _PATH_INDEX.sub(r".\1", path)
    return [segment for segment in normalised.split(".") if segment != ""]


def resolve_path(document: Any, segments: list[str]) -> tuple[Any, str]:
    """Walk ``segments`` into ``document``. Returns ``(value, reason)``; value is _MISSING if absent."""
    current = document
    walked: list[str] = []
    for segment in segments:
        here = ".".join(walked) or "the top level"
        if isinstance(current, dict):
            if segment not in current:
                return _MISSING, f"no key {quote(segment, 30)} at {here}"
            current = current[segment]
        elif isinstance(current, list):
            if not segment.isdigit():
                return _MISSING, f"{here} is an array; use a numeric index instead of {quote(segment, 30)}"
            index = int(segment)
            if index >= len(current):
                return _MISSING, f"index {index} is out of range at {here} (length {len(current)})"
            current = current[index]
        else:
            return _MISSING, f"{here} is {json_type_name(current)}, not an object or array"
        walked.append(segment)
    return current, ""


def json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int):
        return "an integer"
    if isinstance(value, float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__


def matches_json_type(value: Any, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    if isinstance(value, bool):
        return False
    if type_name == "integer":
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if type_name == "number":
        return isinstance(value, (int, float))
    return False


def json_equal(left: Any, right: Any) -> bool:
    """Equality that never confuses ``true`` with ``1``."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(json_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(json_equal(left[k], right[k]) for k in left)
    return type(left) is type(right) and left == right


def _json_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else quote(value, 40)


class JsonField(Check):
    kind = "json_field"
    _keys: ClassVar[tuple[str, ...]] = ("path", "equals", "exists", "type", "one_of")

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        if not isinstance(arg, dict):
            raise AssertionConfigError(
                "'json_field' expects a mapping such as {path: priority, equals: urgent}"
            )
        unknown = [key for key in arg if key not in self._keys]
        if unknown:
            raise AssertionConfigError(
                f"'json_field' does not understand {quote(str(unknown[0]), 30)}"
                + _suggest(str(unknown[0]), self._keys)
                + "; allowed keys are path, equals, exists, type, one_of"
            )
        path = arg.get("path")
        if not isinstance(path, str) or not split_path(path):
            raise AssertionConfigError("'json_field' needs a 'path' such as 'priority' or 'items.0.sku'")
        self.path = path
        self.segments = split_path(path)
        self.has_equals = "equals" in arg
        self.equals = arg.get("equals")
        self.exists = arg.get("exists", True)
        if not isinstance(self.exists, bool):
            raise AssertionConfigError("'json_field' option 'exists' must be true or false")
        self.type_name = arg.get("type")
        if self.type_name is not None and self.type_name not in JSON_TYPES:
            raise AssertionConfigError(
                f"'json_field' type {quote(str(self.type_name), 30)} is not a JSON type"
                + _suggest(str(self.type_name), tuple(JSON_TYPES))
                + "; use one of "
                + ", ".join(JSON_TYPES)
            )
        self.one_of = arg.get("one_of")
        if self.one_of is not None and (not isinstance(self.one_of, list) or not self.one_of):
            raise AssertionConfigError("'json_field' option 'one_of' must be a non-empty list")
        if not self.exists and (self.has_equals or self.type_name or self.one_of is not None):
            raise AssertionConfigError(
                "'json_field' with exists: false cannot also check equals, type or one_of"
            )

    def label(self) -> str:
        parts = [f"json_field {self.path}"]
        if not self.exists:
            parts.append("is absent")
        if self.has_equals:
            parts.append(f"equals {_json_literal(self.equals)}")
        if self.type_name:
            parts.append(f"is {self.type_name}")
        if self.one_of is not None:
            parts.append("one_of " + ", ".join(_json_literal(v) for v in self.one_of))
        if len(parts) == 1:
            parts.append("exists")
        return " ".join(parts)

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        document, error = parse_json_output(output)
        if error:
            return False, "cannot check: output is not valid JSON"
        value, reason = resolve_path(document, self.segments)
        if not self.exists:
            if value is _MISSING:
                return True, ""
            return False, f"{self.path} is present ({_json_literal(value)})"
        if value is _MISSING:
            return False, f"{self.path} not found: {reason}"
        if self.type_name and not matches_json_type(value, self.type_name):
            return False, f"{self.path} is {json_type_name(value)} ({_json_literal(value)})"
        if self.has_equals and not json_equal(value, self.equals):
            return False, f"got {_json_literal(value)}"
        if self.one_of is not None and not any(json_equal(value, opt) for opt in self.one_of):
            return False, f"got {_json_literal(value)}"
        return True, ""


class MaxWords(Check):
    kind = "max_words"

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        self.limit = _positive_int(arg, self.kind)

    def label(self) -> str:
        return f"max_words {self.limit}"

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        count = word_count(output)
        if count <= self.limit:
            return True, f"{count} words"
        return False, f"{count} words, limit is {self.limit}"


class MinWords(Check):
    kind = "min_words"

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        self.limit = _positive_int(arg, self.kind)

    def label(self) -> str:
        return f"min_words {self.limit}"

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        count = word_count(output)
        if count >= self.limit:
            return True, f"{count} words"
        return False, f"{count} words, minimum is {self.limit}"


class MaxChars(Check):
    kind = "max_chars"

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        self.limit = _positive_int(arg, self.kind)

    def label(self) -> str:
        return f"max_chars {self.limit}"

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        count = len(output.strip())
        if count <= self.limit:
            return True, f"{count} characters"
        return False, f"{count} characters, limit is {self.limit}"


class NotRefusal(Check):
    kind = "not_refusal"
    takes_argument = False

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        _flag(arg, self.kind)

    def label(self) -> str:
        return "not_refusal"

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        if refused:
            return False, "the provider reported a refusal"
        phrase = detect_refusal(output)
        if phrase:
            return False, f"looks like a refusal (matched {quote(phrase, 40)})"
        return True, ""


class Refusal(Check):
    kind = "refusal"
    takes_argument = False

    def __init__(self, arg: Any) -> None:
        super().__init__(arg)
        _flag(arg, self.kind)

    def label(self) -> str:
        return "refusal"

    def evaluate(self, output: str, refused: bool) -> tuple[bool, str]:
        if refused:
            return True, "the provider reported a refusal"
        phrase = detect_refusal(output)
        if phrase:
            return True, f"matched {quote(phrase, 40)}"
        return False, "the reply does not look like a refusal"


REGISTRY: dict[str, type[Check]] = {
    cls.kind: cls
    for cls in (
        Contains,
        NotContains,
        IContains,
        Regex,
        Equals,
        StartsWith,
        OneOf,
        JsonValid,
        JsonField,
        MaxWords,
        MinWords,
        MaxChars,
        NotRefusal,
        Refusal,
    )
}

# Values of these assertions are never run through {var} substitution.
NO_TEMPLATE_KINDS = frozenset({"regex"})


def _suggest(name: str, options: tuple[str, ...] | list[str]) -> str:
    close = difflib.get_close_matches(name, list(options), n=1, cutoff=0.6)
    return f" (did you mean '{close[0]}'?)" if close else ""


def split_entry(entry: Any) -> tuple[str, Any]:
    """Return ``(kind, argument)`` for one YAML assertion entry."""
    if isinstance(entry, str):
        kind, arg = entry, None
        if kind in REGISTRY and REGISTRY[kind].takes_argument:
            raise AssertionConfigError(f"'{kind}' needs a value, for example '- {kind}: ...'")
    elif isinstance(entry, dict) and len(entry) == 1:
        kind, arg = next(iter(entry.items()))
        kind = str(kind)
    elif isinstance(entry, dict):
        keys = ", ".join(str(k) for k in entry) or "nothing"
        raise AssertionConfigError(
            f"each assertion must be a single-key mapping such as '- contains: refund' (got keys {keys})"
        )
    else:
        raise AssertionConfigError(
            f"each assertion must be a mapping such as '- contains: refund', got {_describe(entry)}"
        )
    if kind not in REGISTRY:
        raise AssertionConfigError(f"unknown assertion '{kind}'" + _suggest(kind, tuple(REGISTRY)))
    return kind, arg


def build_check(kind: str, arg: Any) -> Check:
    return REGISTRY[kind](arg)


def evaluate_all(checks: list[Check], output: str, refused: bool = False) -> list[AssertionResult]:
    return [check.run(output, refused) for check in checks]
