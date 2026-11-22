from __future__ import annotations

import json

import pytest

from promptcheck.errors import ConfigError, ProviderError
from promptcheck.providers.base import Completion
from promptcheck.providers.replay import FORMAT, RecordingProvider, ReplayProvider, load_recordings
from promptcheck.runner import run_suite
from promptcheck.suite import load_suite

from .conftest import EXAMPLE_RECORDINGS, EXAMPLE_SUITE, FakeProvider
from .test_providers import request


def write(tmp_path, data, name="rec.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_shorthand_format(tmp_path):
    path = write(tmp_path, {"triage": "one", "multi": ["a", "b"]})
    provider = ReplayProvider(path)
    assert provider.complete(request()).text == "one"
    assert provider.complete(request(case_id="multi", sample=1)).text == "b"
    assert provider.label == f"replay ({path.as_posix()})"


def test_full_format_with_metadata(tmp_path):
    req = request()
    path = write(
        tmp_path,
        {
            "format": FORMAT,
            "cases": {
                "triage": [
                    {"output": "x", "model": "m-1", "request_hash": req.fingerprint(), "refused": True}
                ]
            },
        },
    )
    completion = ReplayProvider(path).complete(req)
    assert (completion.text, completion.model, completion.refused) == ("x", "m-1", True)
    assert completion.warnings == []


def test_stale_recording_warns(tmp_path):
    path = write(
        tmp_path, {"format": FORMAT, "cases": {"triage": [{"output": "x", "request_hash": "0" * 16}]}}
    )
    completion = ReplayProvider(path).complete(request())
    assert completion.warnings and completion.warnings[0].startswith("stale recording")


def test_missing_case_and_sample(tmp_path):
    provider = ReplayProvider(write(tmp_path, {"triage": ["only one"]}))
    with pytest.raises(ProviderError, match="no recording for case 'other'"):
        provider.complete(request(case_id="other"))
    with pytest.raises(ProviderError, match=r"has 1 recorded output\(s\) but sample #2 was requested"):
        provider.complete(request(sample=1))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "expected a JSON object"),
        (json.dumps({"format": "other/9", "cases": {}}), "unsupported recording format"),
        (json.dumps({"format": FORMAT, "cases": []}), "'cases' must map case ids"),
        (json.dumps({"triage": [{"no_output": 1}]}), "case 'triage', sample #1"),
    ],
)
def test_invalid_files(tmp_path, content, message):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_recordings(path)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="replay file not found"):
        ReplayProvider(tmp_path / "missing.json")


def test_example_recordings_cover_the_example_suite_without_staleness():
    suite = load_suite(EXAMPLE_SUITE)
    run = run_suite(suite, ReplayProvider(EXAMPLE_RECORDINGS))
    assert not any(c.warnings for c in run.cases), "example recordings must match the suite"
    statuses = {c.case.id: c.status for c in run.cases}
    assert statuses == {
        "triage-double-charge": "pass",
        "triage-checkout-outage": "fail",
        "triage-feature-request": "fail",
        "triage-locked-account": "pass",
        "triage-ambiguous-ticket": "pass",
        "summarise-refund-thread": "pass",
        "summarise-escalation-thread": "fail",
        "reply-admin-password-reset": "pass",
        "reply-prompt-injection": "pass",
    }
    assert run.flaky == 1


def test_record_then_replay_round_trip(tmp_path, make_suite):
    suite = make_suite(
        """
        name: Round trip
        defaults: {model: m}
        cases:
          - {id: a, prompt: "hi", assert: [{equals: hello}]}
          - {id: b, prompt: "count", repeat: 2, assert: [{min_words: 1}]}
          - {id: c, prompt: "boom", assert: [json_valid]}
        """
    )
    live = FakeProvider(
        {"a": "hello", "b": ["one", "two"], "c": Completion(text="", refused=True)},
        raise_for={"c": RuntimeError("down")},
    )
    recorder = RecordingProvider(live)
    run = run_suite(suite, recorder)
    path = tmp_path / "out" / "rec.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"format": FORMAT, "cases": {"old": ["kept"], "a": ["stale"]}}))
    recorded, skipped = recorder.save(
        path, suite_name=suite.name, expected={c.case.id: len(c.samples) for c in run.cases}
    )
    assert recorded == ["a", "b"]
    assert skipped == ["c"]
    document = json.loads(path.read_text())
    assert document["format"] == FORMAT
    assert document["suite"] == "Round trip"
    assert document["cases"]["old"] == ["kept"], "cases that did not run are preserved"
    assert [e["output"] for e in document["cases"]["b"]] == ["one", "two"]

    replayed = run_suite(suite, ReplayProvider(path), cases=suite.cases[:2])
    assert [c.status for c in replayed.cases] == ["pass", "pass"]
    assert not any(c.warnings for c in replayed.cases)
