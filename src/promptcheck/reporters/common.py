"""Formatting helpers shared by reporters."""

from __future__ import annotations

import re

from promptcheck.runner import CaseResult, RunResult

# C0/C1 control characters (except tab and newline), DEL, and bidirectional overrides.
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def clean_terminal(text: str) -> str:
    """Make untrusted text (model output) safe to print to a terminal.

    Escape sequences in model output could otherwise recolour the terminal, rewrite
    earlier lines (carriage returns), change the window title or reorder text.
    Each control character is shown as a visible escape such as ``\\x1b``.
    """

    def visible(match: re.Match[str]) -> str:
        code = ord(match.group(0))
        return f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}"

    return _CONTROL.sub(visible, text)


def percent(value: float) -> str:
    """``0.6667 -> '66.7%'``, ``1.0 -> '100%'``."""
    text = f"{value * 100:.1f}".rstrip("0").rstrip(".")
    return f"{text}%"


def duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)} min {rest:.0f} s"


def latency(ms: float) -> str:
    """Format a latency; sub-millisecond values (replay) are not worth showing."""
    if ms < 1:
        return ""
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def samples_text(case: CaseResult) -> str:
    return f"{case.passed_samples}/{len(case.samples)}"


def rate_note(case: CaseResult) -> str:
    """Explains a repeated case's pass rate against its threshold, e.g. ``80% (needs 80%)``."""
    if len(case.samples) <= 1:
        return ""
    return f"{percent(case.pass_rate)} (needs {percent(case.threshold)})"


def summary_line(run: RunResult) -> str:
    parts = [f"{run.passed} passed", f"{run.failed} failed"]
    if run.errored:
        parts.append(f"{run.errored} errored")
    if run.flaky:
        parts.append(f"{run.flaky} flaky")
    return ", ".join(parts)


def excerpt(text: str, limit: int = 280) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
