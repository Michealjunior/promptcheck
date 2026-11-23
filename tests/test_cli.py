from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from promptcheck import __version__
from promptcheck.cli import main
from promptcheck.providers import http

from .conftest import EXAMPLE_RECORDINGS, EXAMPLE_SUITE, ROOT, FakeOpener, FakeProvider, StubServer

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
    monkeypatch.setattr(http, "urlopen", opener)
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
    assert "does-not-exist.yaml: suite file not found" in err


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


# ----------------------------------------------------------------- regression tests (QA)

CHAT_OK = {"choices": [{"message": {"content": "hello there"}, "finish_reason": "stop"}]}


def _chat(server: StubServer, content: str = "hello there", delay: float = 0.0) -> None:
    def respond(handler, path, body):
        if delay:
            time.sleep(delay)
        payload = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        StubServer.reply_json(handler, path, body, payload=payload)

    server.respond = respond


def _suite(tmp_path, cases: int = 1) -> str:
    path = tmp_path / "suite.yaml"
    body = "".join(f"  - {{id: c{i}, prompt: hi, assert: [{{contains: hello}}]}}\n" for i in range(cases))
    path.write_text(f"name: Stubbed\ndefaults: {{model: stub-model}}\ncases:\n{body}", encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("base_url", ["", "  ", "api.openai.com/v1", "ftp://example.test/v1"])
def test_empty_or_invalid_base_url_is_rejected_before_any_request(base_url, monkeypatch):
    # M7: --base-url "" used to fall back to https://api.openai.com/v1 and send the key there.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-cli-test-000")
    opener = FakeOpener()
    monkeypatch.setattr(http, "urlopen", opener)
    code, out, err = cli("run", EXAMPLE_SUITE, "--base-url", base_url)
    assert code == 2
    assert "--base-url must" in err
    assert opener.requests == []
    assert out == ""


def test_live_run_against_local_stub_server(tmp_path, monkeypatch, stub_server):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-cli-test-000")
    server = stub_server()
    _chat(server)
    code, _out, err = cli("run", _suite(tmp_path, 2), "--base-url", f"{server.url}/v1", "--no-color")
    assert code == 0, err
    assert len(server.requests) == 2
    method, path, headers, body = server.requests[0]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert headers["Authorization"] == "Bearer sk-cli-test-000"
    assert body["model"] == "stub-model"


@pytest.mark.parametrize(
    ("flag", "make_path", "message"),
    [
        ("--junit", lambda tmp: tmp, "is a directory, cannot write the junit report"),
        (
            "--html",
            lambda tmp: tmp / "afile" / "report.html",
            "is not a directory, cannot write the html report",
        ),
        ("--markdown", lambda tmp: tmp / "suite.yaml", "is the same file as the suite file"),
    ],
)
def test_unwritable_outputs_fail_before_the_run(tmp_path, monkeypatch, flag, make_path, message):
    # M8: an unwritable path used to fail with a traceback only after the whole run.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-cli-test-000")
    opener = FakeOpener()
    monkeypatch.setattr(http, "urlopen", opener)
    suite = _suite(tmp_path)
    (tmp_path / "afile").write_text("not a directory")
    code, _out, err = cli("run", suite, "--base-url", "http://127.0.0.1:9/v1", flag, make_path(tmp_path))
    assert code == 2
    assert message in err
    assert opener.requests == [], "nothing is sent when an output cannot be written"


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions as a normal user")
def test_read_only_output_directory_fails_before_the_run(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        code, _, err = cli("run", EXAMPLE_SUITE, *REPLAY, "--junit", locked / "junit.xml")
    finally:
        locked.chmod(0o700)
    assert code == 2
    assert "permission denied" in err


def test_write_failure_after_the_run_keeps_the_summary_and_has_no_traceback(tmp_path, monkeypatch):
    def broken(run, path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("promptcheck.cli.write_junit", broken)
    html = tmp_path / "report.html"
    code, out, err = cli("run", EXAMPLE_SUITE, *REPLAY, "--junit", tmp_path / "j.xml", "--html", html)
    assert code == 2
    assert "FAILED  6 passed, 3 failed, 1 flaky" in out, "the summary is printed before the failure"
    assert f"could not write the junit report to {tmp_path / 'j.xml'}: Permission denied" in err
    assert "Traceback" not in err
    assert html.exists(), "other reports are still written"


def test_failed_recording_write_rescues_the_outputs(tmp_path, monkeypatch, stub_server):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-cli-test-000")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    server = stub_server()
    _chat(server)
    target = tmp_path / "recordings.json"
    from promptcheck.providers import replay

    real_write = replay.atomic_write_text

    def flaky_write(path, text):
        if Path(path) == target:
            raise PermissionError(13, "Permission denied")
        return real_write(path, text)

    monkeypatch.setattr(replay, "atomic_write_text", flaky_write)
    code, out, err = cli("run", _suite(tmp_path), "--base-url", f"{server.url}/v1", "--record", target)
    assert code == 2
    assert "PASSED  1 passed" in out
    assert f"could not write the recording to {target}: Permission denied" in err
    rescue = Path(err.split("the outputs were saved to ")[1].split(" instead")[0])
    assert json.loads(rescue.read_text())["cases"]["c0"][0]["output"] == "hello there"


def test_record_refuses_a_conflicted_file_before_sending_anything(tmp_path, monkeypatch):
    # M5 at the CLI level: the file is left untouched and no request is made.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-cli-test-000")
    opener = FakeOpener()
    monkeypatch.setattr(http, "urlopen", opener)
    target = tmp_path / "recordings.json"
    target.write_text('{\n<<<<<<< HEAD\n"a": "x"\n=======\n"a": "y"\n>>>>>>> b\n}\n')
    before = target.read_text()
    code, _, err = cli("run", _suite(tmp_path), "--base-url", "http://127.0.0.1:9/v1", "--record", target)
    assert code == 2
    assert "refusing to overwrite" in err and "merge-conflict markers" in err
    assert target.read_text() == before
    assert opener.requests == []


@pytest.mark.parametrize(
    "args",
    [
        ["--repeat", "101"],
        ["--repeat", "0"],
        ["--retries", "11"],
        ["--retries", "1000000"],
        ["--concurrency", "65"],
        ["--timeout", "0"],
        ["--timeout", "3601"],
        ["--timeout", "nan"],
        ["--min-pass-rate", "0"],
        ["--min-pass-rate", "0%"],
        ["--model", ""],
    ],
)
def test_out_of_range_options_are_rejected(args, capsys):
    assert main(["run", str(EXAMPLE_SUITE), *REPLAY, *args]) == 2
    assert "error: argument" in capsys.readouterr().err


def test_internal_errors_exit_2_and_are_reported_as_bugs(monkeypatch):
    def broken_provider(*args, **kwargs):
        return FakeProvider({}, raise_for={c: RuntimeError("boom") for c in ("triage-double-charge",)})

    monkeypatch.setattr("promptcheck.cli.create_provider", broken_provider)
    code, _out, err = cli("run", EXAMPLE_SUITE, "--case", "triage-double-charge", "--no-color")
    assert code == 2
    assert "internal error: RuntimeError: boom (case 'triage-double-charge', sample 1)" in err
    assert "this is a bug in promptcheck, not a failure of your prompt" in err


def test_interrupted_run_exits_130_with_a_short_summary(monkeypatch):
    from promptcheck.runner import RunInterrupted

    def interrupted(*args, **kwargs):
        raise RunInterrupted(3, 13)

    monkeypatch.setattr("promptcheck.cli.run_suite", interrupted)
    code, out, _ = cli("run", EXAMPLE_SUITE, *REPLAY, "--no-color")
    assert code == 130
    assert "INTERRUPTED  after 3 of 13 samples; no reports or recordings were written" in out


@pytest.mark.skipif(os.name == "nt", reason="SIGINT delivery differs on Windows")
def test_ctrl_c_stops_queued_requests_and_exits_promptly(tmp_path, stub_server):
    # M6: Ctrl-C used to send every queued request and take 13.6 s to exit.
    server = stub_server()
    _chat(server, delay=3)
    env = {"PATH": os.environ.get("PATH", ""), "OPENAI_API_KEY": "sk-cli-test-000", "NO_COLOR": "1"}
    command = [sys.executable, "-m", "promptcheck", "run", _suite(tmp_path, 12)]
    command += ["--base-url", f"{server.url}/v1", "-j", "2", "--retries", "0"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=ROOT)
    deadline = time.monotonic() + 10
    while len(server.requests) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(server.requests) == 2, "two requests should be in flight"
    process.send_signal(signal.SIGINT)
    interrupted_at = time.monotonic()
    out, err = process.communicate(timeout=15)
    assert process.returncode == 130, err.decode()
    assert time.monotonic() - interrupted_at < 2
    assert b"INTERRUPTED  after 0 of 12 samples" in out
    time.sleep(1)
    assert len(server.requests) == 2, "queued requests must never be sent"
