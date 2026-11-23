from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import pytest

from promptcheck.providers.replay import ReplayProvider
from promptcheck.reporters import (
    TerminalReporter,
    render_html,
    render_junit,
    render_markdown,
    use_color,
    write_html,
    write_junit,
    write_markdown,
)
from promptcheck.runner import RunResult, run_suite
from promptcheck.suite import load_suite

from .conftest import EXAMPLE_RECORDINGS, EXAMPLE_SUITE, FakeProvider


@pytest.fixture(scope="module")
def example_run() -> RunResult:
    return run_suite(load_suite(EXAMPLE_SUITE), ReplayProvider(EXAMPLE_RECORDINGS))


@pytest.fixture
def hostile_run(make_suite) -> RunResult:
    """Outputs with characters that break naive XML, HTML and Markdown writers."""
    suite = make_suite(
        """
        name: Hostile <suite> & "friends"
        cases:
          - id: html
            prompt: hi
            assert: [{equals: safe}]
          - id: control
            prompt: hi
            assert: [json_valid]
          - id: ok
            prompt: hi
            assert: [{contains: fine}]
        """
    )
    outputs = {
        "html": "<script>alert('x')</script> | pipe ```fence```",
        "control": "bad\x00byte\x1b[31m",
        "ok": "fine",
    }
    return run_suite(suite, FakeProvider(outputs))


# ----------------------------------------------------------------- JUnit


def test_junit_structure(example_run):
    root = ET.fromstring(render_junit(example_run))
    assert root.tag == "testsuites"
    assert root.attrib["tests"] == "9"
    assert root.attrib["failures"] == "3"
    assert root.attrib["errors"] == "0"
    (suite,) = root.findall("testsuite")
    assert suite.attrib["name"] == "Helpdesk assistant"
    cases = suite.findall("testcase")
    assert [c.attrib["name"] for c in cases][:2] == ["triage-double-charge", "triage-checkout-outage"]
    assert {c.attrib["classname"] for c in cases} == {"promptcheck.helpdesk-assistant"}
    outage = next(c for c in cases if c.attrib["name"] == "triage-checkout-outage")
    failure = outage.find("failure")
    assert failure is not None
    assert failure.attrib["message"] == 'json_field priority equals "urgent": got "high"'
    assert "sample 1: FAILED json_field priority equals" in failure.text
    assert '"priority": "high"' in outage.find("system-out").text
    passing = next(c for c in cases if c.attrib["name"] == "triage-double-charge")
    assert passing.find("failure") is None
    props = {p.attrib["name"]: p.attrib["value"] for p in suite.find("properties")}
    assert props["model"] == "gpt-4.1-mini"
    assert props["provider"].startswith("replay")


def test_junit_survives_control_characters(hostile_run, tmp_path):
    path = write_junit(hostile_run, tmp_path / "nested" / "junit.xml")
    root = ET.parse(path).getroot()
    out = root.find(".//testcase[@name='control']/system-out").text
    assert "\x00" not in out and "bad" in out


def test_junit_errors(make_suite):
    from promptcheck.errors import ProviderError

    suite = make_suite("name: E\ncases:\n  - {id: e, prompt: hi, assert: [json_valid]}\n")
    run = run_suite(suite, FakeProvider({}, raise_for={"e": ProviderError("HTTP 503: overloaded")}))
    root = ET.fromstring(render_junit(run))
    assert root.attrib["errors"] == "1"
    error = root.find(".//testcase/error")
    assert error.attrib["message"] == "error: HTTP 503: overloaded"


# ----------------------------------------------------------------- Markdown


def test_markdown_summary(example_run):
    md = render_markdown(example_run)
    assert md.startswith("## ❌ promptcheck failed: Helpdesk assistant")
    assert "**6/9 cases passed (66.7%)**" in md
    assert "| ✅ | `triage-double-charge` | 1/1 | all assertions passed |" in md
    assert '| ❌ | `triage-checkout-outage` | 0/1 | json\\_field priority equals "urgent": got "high" |' in md
    assert "| ✅ | `triage-ambiguous-ticket` | 4/5 (80%) | flaky but within threshold |" in md
    assert "<details>" in md and "Details for 3 failing case(s)" in md
    assert "104 words, limit is 60" in md


def test_markdown_escapes_tables_and_fences(hostile_run, tmp_path):
    md = write_markdown(hostile_run, tmp_path / "s.md").read_text()
    row = next(line for line in md.splitlines() if line.startswith("| ❌ | `html`"))
    assert "\\|" in row
    assert row.count(" | ") == 3, "pipes inside cells must not create extra columns"
    assert "````\n<script>" in md, "a longer fence wraps output that contains ```"


def test_markdown_pass_and_threshold(example_run):
    md = render_markdown(example_run, min_pass_rate=0.6)
    assert md.startswith("## ✅ promptcheck passed")
    assert "Minimum case pass rate: 60%." in md


# ----------------------------------------------------------------- HTML


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)

    def handle_data(self, data):
        self.text.append(data)


def test_html_report_contents(example_run):
    page = render_html(example_run)
    assert page.startswith("<!doctype html>")
    parser = _Collector()
    parser.feed(page)
    assert parser.tags.count("details") >= 9
    text = " ".join(parser.text)
    for case in example_run.cases:
        assert case.case.id in text
    assert "3 cases failing" in text
    assert "6 of 9 cases passed" in text
    assert 'json_field priority equals "urgent"' in page.replace("&quot;", '"')
    assert '<meta name="description"' in page
    assert 'property="og:title"' in page
    assert 'rel="icon"' in page
    assert "prefers-reduced-motion" in page
    assert 'data-status="fail" open' in page
    assert len(re.findall(r'class="dot dot--(pass|fail)"', page)) == 5


def test_html_escapes_model_output(hostile_run, tmp_path):
    page = write_html(hostile_run, tmp_path / "r.html").read_text()
    assert "<script>alert" not in page
    assert "&lt;script&gt;alert" in page
    assert "Hostile &lt;suite&gt; &amp; &quot;friends&quot;" in page


def test_html_all_passing(make_suite):
    suite = make_suite("name: Green\ncases:\n  - {id: a, prompt: hi, assert: [{contains: ok}]}\n")
    page = render_html(run_suite(suite, FakeProvider({"a": "ok"})))
    assert "All 1 cases passing" in page
    assert "verdict--pass" in page


# ----------------------------------------------------------------- terminal


def test_terminal_plain_output(example_run):
    stream = io.StringIO()
    reporter = TerminalReporter(stream, color=False)
    reporter.report(example_run, {"html": "reports/report.html"})
    out = stream.getvalue()
    assert "\033[" not in out
    assert re.search(r"^PASS triage-double-charge\s+1/1", out, re.M)
    assert re.search(r"^FAIL triage-checkout-outage\s+0/1", out, re.M)
    assert '✗ json_field priority equals "urgent": got "high"' in out
    assert "4/5  80% (needs 80%)" in out
    assert "sample 3: json_field category one_of" in out
    assert "FAILED  6 passed, 3 failed, 1 flaky" in out
    assert "6/9 cases (66.7%)" in out
    assert "reports/report.html" in out


def test_terminal_header_pluralises():
    stream = io.StringIO()
    reporter = TerminalReporter(stream, color=False)
    reporter.header("Suite", "replay", None, 1, 1)
    reporter.header("Suite", "replay", "m", 2, 5)
    out = stream.getvalue()
    assert "1 case, 1 sample\n" in out
    assert "2 cases, 5 samples\n" in out


def test_terminal_colour_and_quiet(example_run):
    coloured = io.StringIO()
    TerminalReporter(coloured, color=True).report(example_run)
    assert "\033[1;37;41m FAIL \033[0m" in coloured.getvalue()
    quiet = io.StringIO()
    TerminalReporter(quiet, color=False, quiet=True).report(example_run)
    assert quiet.getvalue().count("\n") == 1


def test_terminal_ascii_fallback(example_run):
    class Latin1(io.StringIO):
        encoding = "latin-1"

    stream = Latin1()
    TerminalReporter(stream, color=False).report(example_run)
    stream.getvalue().encode("latin-1")


def test_terminal_verbose_shows_passing_assertions(example_run):
    stream = io.StringIO()
    TerminalReporter(stream, color=False, verbose=True).report(example_run)
    assert '✓ json_field category equals "billing"' in stream.getvalue()


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("environ", "tty", "expected"),
    [
        ({}, True, True),
        ({}, False, False),
        ({"NO_COLOR": "1"}, True, False),
        ({"NO_COLOR": ""}, True, True),
        ({"FORCE_COLOR": "1"}, False, True),
        ({"TERM": "dumb"}, True, False),
    ],
)
def test_use_color(environ, tty, expected):
    stream = _Tty() if tty else io.StringIO()
    assert use_color(stream, environ=environ) is expected


def test_use_color_flag_wins():
    assert use_color(_Tty(), force=False, environ={}) is False


# ----------------------------------------------------------------- regression tests (QA)

ESCAPES = "\x1b]0;pwned-title\x07\x1b[2J\x1b[31mRED\rOVERWRITE \u202eevil"


@pytest.fixture
def injection_run(make_suite) -> RunResult:
    suite = make_suite(
        """
        name: Injection
        cases:
          - id: terminal
            prompt: hi
            assert: [{equals: positive}]
          - id: comment
            prompt: hi
            assert: [{equals: positive}]
          - id: "tick`id|pipe"
            prompt: hi
            assert: [{contains: hello}]
        """
    )
    outputs = {
        "terminal": ESCAPES,
        "comment": "<!-- hidden <img src=x onerror=alert(1)> **bold**",
        "tick`id|pipe": "nope",
    }
    return run_suite(suite, FakeProvider(outputs))


@pytest.mark.parametrize("color", [False, True])
def test_terminal_escapes_model_control_characters(injection_run, color):
    stream = io.StringIO()
    TerminalReporter(stream, color=color).report(injection_run)
    out = stream.getvalue()
    for raw in ("\x1b]0;", "\x1b[2J", "\x1b[31mRED", "\x07", "\r", "\u202e"):
        assert raw not in out, f"raw control sequence {raw!r} reached the terminal"
    assert "\\x1b]0;pwned-title\\x07" in out
    assert "\\x0dOVERWRITE" in out
    assert "\\u202eevil" in out


def test_markdown_escapes_html_and_markdown_from_model_output(injection_run):
    md = render_markdown(injection_run)
    outside_code = re.sub(r"(?ms)^(`{3,})\n.*?^\1$", "", md)
    assert "<!--" not in outside_code, "an HTML comment would hide the rest of the PR comment"
    assert "<img" not in outside_code
    assert "&lt;\\!--" in md, "rendered as the literal text <!--"
    assert "\\*\\*bold\\*\\*" in md
    assert "\x1b" not in md
    row = next(line for line in md.splitlines() if "tick" in line and line.startswith("|"))
    assert "``tick`id\\|pipe``" in row, "ids with backticks or pipes stay inside one code span and cell"
    assert len(re.findall(r"(?<!\\)\|", row)) == 5, "the table row keeps exactly four cells"
    assert "#### ``tick`id|pipe``" in md


def test_html_report_escapes_injection(injection_run):
    page = render_html(injection_run)
    assert "<!-- hidden" not in page
    assert "&lt;!-- hidden" in page
    assert "<img src=x" not in page
