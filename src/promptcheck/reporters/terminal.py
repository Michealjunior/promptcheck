"""Human-readable terminal summary, coloured when the stream is a TTY."""

from __future__ import annotations

import os
import sys
from typing import TextIO

from promptcheck import __version__
from promptcheck.reporters.common import duration, excerpt, latency, percent, rate_note, summary_line
from promptcheck.runner import ERROR, FAIL, PASS, CaseResult, RunResult

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
    "pass_badge": "1;30;42",
    "fail_badge": "1;37;41",
    "error_badge": "1;30;43",
}


def use_color(stream: TextIO, force: bool | None = None, environ: dict[str, str] | None = None) -> bool:
    """Colour only for TTYs; NO_COLOR disables, FORCE_COLOR enables (e.g. in CI logs)."""
    env = os.environ if environ is None else environ
    if force is not None:
        return force
    if env.get("NO_COLOR"):
        return False
    if env.get("FORCE_COLOR"):
        return True
    if env.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _encoding(stream: TextIO) -> str:
    # In-memory streams report no encoding; they accept any text.
    return getattr(stream, "encoding", None) or "utf-8"


def _supports_unicode(stream: TextIO) -> bool:
    try:
        "✓✗·─…│".encode(_encoding(stream))
    except (LookupError, UnicodeEncodeError):
        return False
    return True


class TerminalReporter:
    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        color: bool | None = None,
        verbose: bool = False,
        quiet: bool = False,
        max_failed_samples: int = 3,
    ) -> None:
        self.stream = stream or sys.stdout
        self.color = use_color(self.stream, color)
        self.unicode = _supports_unicode(self.stream)
        self.verbose = verbose
        self.quiet = quiet
        self.max_failed_samples = max_failed_samples

    # -- styling -----------------------------------------------------------------

    def style(self, text: str, *names: str) -> str:
        if not self.color or not names:
            return text
        codes = ";".join(_CODES[n] for n in names)
        return f"\033[{codes}m{text}\033[0m"

    def sym(self, name: str) -> str:
        table = {
            "ok": ("✓", "+"),
            "bad": ("✗", "x"),
            "dot": ("·", "-"),
            "rule": ("─", "-"),
            "warn": ("!", "!"),
        }
        fancy, plain = table[name]
        return fancy if self.unicode else plain

    def write(self, line: str = "") -> None:
        if not self.unicode:
            # Legacy consoles: replace anything (including model output) they cannot encode.
            encoding = _encoding(self.stream)
            try:
                line = line.encode(encoding, "replace").decode(encoding)
            except LookupError:
                line = line.encode("ascii", "replace").decode("ascii")
        self.stream.write(line + "\n")

    # -- report --------------------------------------------------------------------

    def header(self, run_suite_name: str, provider: str, model: str | None, cases: int, samples: int) -> None:
        if self.quiet:
            return
        dot = f" {self.sym('dot')} "
        title = self.style(f"promptcheck {__version__}", "bold")
        meta = dot.join(x for x in (run_suite_name, provider, model) if x)
        self.write(f"{title}{dot}{meta}")
        noun = "case" if cases == 1 else "cases"
        self.write(self.style(f"{cases} {noun}, {samples} samples", "dim"))
        self.write()

    def report(
        self, run: RunResult, outputs: dict[str, str] | None = None, min_pass_rate: float = 1.0
    ) -> None:
        if not self.quiet:
            width = max([len(c.case.id) for c in run.cases] + [10])
            for case in run.cases:
                self._case(case, width)
            for notice in run.notices:
                self.write(self.style(f"{self.sym('warn')} note: {notice}", "yellow"))
            self.write(self.style(self.sym("rule") * 60, "dim"))
        self._summary(run, min_pass_rate)
        if outputs and not self.quiet:
            for label, path in outputs.items():
                self.write(self.style(f"  {label:<9}{path}", "dim"))

    def _badge(self, status: str) -> str:
        text = {PASS: " PASS ", FAIL: " FAIL ", ERROR: " ERR  "}[status]
        if self.color:
            return self.style(text, f"{status}_badge")
        return text.strip().ljust(4)

    def _case(self, case: CaseResult, width: int) -> None:
        counts = f"{case.passed_samples}/{len(case.samples)}"
        note = rate_note(case)
        timing = latency(case.latency_ms)
        tail = "  ".join(x for x in (counts, note, timing) if x)
        self.write(f"{self._badge(case.status)} {case.case.id:<{width}}  {self.style(tail, 'dim')}")
        indent = " " * (7 if self.color else 5)
        for warning in case.warnings:
            self.write(indent + self.style(f"{self.sym('warn')} {warning}", "yellow"))
        failing = [s for s in case.samples if s.status != PASS]
        if case.passed and not failing and not self.verbose:
            return
        shown = 0
        for sample in case.samples:
            if sample.status == PASS and not self.verbose:
                continue
            if shown >= self.max_failed_samples and not self.verbose:
                remaining = len(failing) - shown
                self.write(indent + self.style(f"… {remaining} more failing sample(s)", "dim"))
                break
            shown += 1
            prefix = f"sample {sample.index + 1}: " if len(case.samples) > 1 else ""
            if sample.error:
                self.write(indent + self.style(f"{self.sym('bad')} {prefix}error: {sample.error}", "yellow"))
                continue
            for result in sample.assertions:
                if result.passed and not self.verbose:
                    continue
                mark = (
                    self.style(self.sym("ok"), "green")
                    if result.passed
                    else self.style(self.sym("bad"), "red")
                )
                detail = f": {result.detail}" if result.detail else ""
                if result.passed:
                    detail = self.style(detail, "dim")
                self.write(f"{indent}{mark} {prefix}{result.label}{detail}")
            if sample.output is not None and (sample.status != PASS or self.verbose):
                text = sample.output if self.verbose else excerpt(sample.output)
                for line in (text or "(empty output)").splitlines() or [""]:
                    self.write(
                        indent + self.style("│ " if self.unicode else "| ", "dim") + self.style(line, "dim")
                    )

    def _summary(self, run: RunResult, min_pass_rate: float) -> None:
        verdict_ok = run.ok(min_pass_rate)
        verdict = self.style("PASSED", "bold", "green") if verdict_ok else self.style("FAILED", "bold", "red")
        rate = f"{run.passed}/{run.total} cases ({percent(run.pass_rate)})"
        if min_pass_rate < 1:
            rate += f", needs {percent(min_pass_rate)}"
        self.write(
            f"{verdict}  {summary_line(run)}  {self.style(self.sym('dot'), 'dim')}  "
            f"{rate}  {self.style(self.sym('dot'), 'dim')}  {duration(run.duration_s)}"
        )
