from __future__ import annotations

import io
import json
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

import pytest

from promptcheck import __version__
from promptcheck.cli import main

from .conftest import EXAMPLE_RECORDINGS, EXAMPLE_SUITE, ROOT, FakeOpener

REPLAY = ["--provider", "replay", "--replay", str(EXAMPLE_RECORDINGS)]


def cli(*args):
    out, err = io.StringIO(), io.StringIO()
    code = main([str(a) for a in args], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "FORCE_COLOR"):
        monkeypatch.delenv(name, raising=False)


def test_example_suite_fails_with_exit_code_1_and_writes_reports(tmp_path):
    junit, md, html = tmp_path / "junit.xml", tmp_path / "summary.md", tmp_path / "report.html"
    code, out, err = cli("run", EXAMPLE_SUITE, *REPLAY, "--junit", junit, "--markdown", md, "--html", html)
    assert code == 1, err
    assert "FAILED  6 passed, 3 failed, 1 flaky" in out
    assert ET.parse(junit).getroot().attrib["failures"] == "3"
    assert "promptcheck failed" in md.read_text()
    assert "<!doctype html>" in html.read_text()
    assert str(junit) in out


def test_min_pass_rate_controls_exit_code():
    assert cli("run", EXAMPLE_SUITE, *REPLAY, "--min-pass-rate", "0.6", "-q")[0] == 0
    assert cli("run", EXAMPLE_SUITE, *REPLAY, "--min-pass-rate", "66%", "-q")[0] == 0
    assert cli("run", EXAMPLE_SUITE, *REPLAY, "--min-pass-rate", "0.7", "-q")[0] == 1


def test_case_filter_selects_passing_cases():
    code, out, _ = cli("run", EXAMPLE_SUITE, *REPLAY, "--case", "reply-*", "--case", "triage-double-*")
    assert code == 0
    assert "PASSED  3 passed, 0 failed" in out
    assert "triage-checkout-outage" not in out


def test_unknown_case_pattern_is_a_usage_error():
    code, _, err = cli("run", EXAMPLE_SUITE, *REPLAY, "--case", "nope*")
    assert code == 2
    assert "no case in" in err and "matches --case nope*" in err


def test_replay_shorter_than_repeat_errors_per_case():
    code, out, _ = cli("run", EXAMPLE_SUITE, *REPLAY, "--case", "triage-double-charge", "--repeat", "2")
    assert code == 1
    assert "has 1 recorded output(s) but sample #2 was requested" in out


def test_validate(tmp_path):
    code, out, _ = cli("validate", EXAMPLE_SUITE)
    assert code == 0
    assert "ok · 9 cases, 13 samples, 35 assertions" in out
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: Bad\ncases:\n  - id: x\n    prompt: hi\n    assert: [{max_words: lots}]\n")
    code, _, err = cli("validate", bad)
    assert code == 2
    assert "case 'x': assertion #1: 'max_words' expects a positive integer" in err


def test_missing_api_key_is_a_usage_error_without_leaking_anything():
    code, out, err = cli("run", EXAMPLE_SUITE)
    assert code == 2
    assert "OPENAI_API_KEY is not set" in err
    assert out == ""


def test_record_with_mocked_live_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-cli-test-000")
    reply = {"choices": [{"message": {"content": "I can't help with that."}, "finish_reason": "stop"}]}
    opener = FakeOpener(reply)
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    recording = tmp_path / "rec.json"
    code, out, err = cli("run", EXAMPLE_SUITE, "--case", "reply-prompt-injection", "--record", recording)
    assert code == 1, err  # passes refusal but fails icontains "Ledgerly"
    assert "sk-cli-test-000" not in out + err
    document = json.loads(recording.read_text())
    assert document["cases"]["reply-prompt-injection"][0]["output"] == "I can't help with that."
    assert opener.body()["model"] == "gpt-4.1-mini"
    code, out, _ = cli(
        "run", EXAMPLE_SUITE, "--provider", "replay", "--replay", recording, "--case", "reply-*n"
    )
    assert "stale" not in out


def test_record_and_replay_together_is_rejected(tmp_path):
    code, _, err = cli("run", EXAMPLE_SUITE, *REPLAY, "--record", tmp_path / "x.json")
    assert code == 2
    assert "--record needs a live provider" in err


@pytest.mark.parametrize(
    "args",
    [
        ["run"],
        ["run", "x.yaml", "--min-pass-rate", "2"],
        ["run", "x.yaml", "--concurrency", "0"],
        ["run", "x.yaml", "--provider", "nope"],
    ],
)
def test_argument_errors_exit_2(args, capsys):
    assert main(args) == 2


def test_missing_suite_file():
    code, _, err = cli("run", "does-not-exist.yaml", *REPLAY)
    assert code == 2
    assert "does-not-exist.yaml: file not found" in err


def test_no_command_prints_help():
    code, out, _ = cli()
    assert code == 2
    assert "usage: promptcheck" in out


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"promptcheck {__version__}"


def test_module_entry_point_and_no_color_env():
    result = subprocess.run(
        [sys.executable, "-m", "promptcheck", "run", str(EXAMPLE_SUITE), *REPLAY],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"NO_COLOR": "1", "PATH": ""},
        check=False,
    )
    assert result.returncode == 1
    assert "\033[" not in result.stdout
    assert "Helpdesk assistant" in result.stdout
