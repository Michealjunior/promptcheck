from __future__ import annotations

import dataclasses
import threading
import time

import pytest

from promptcheck.errors import ProviderError
from promptcheck.providers.base import Completion, CompletionRequest, Provider
from promptcheck.runner import ERROR, FAIL, PASS, RunInterrupted, run_suite

from .conftest import FakeProvider

SUITE = """
name: Runner
defaults:
  model: test-model
  timeout: 12
cases:
  - id: good
    prompt: "say yes"
    assert:
      - equals: "yes"
  - id: bad
    prompt: "say yes"
    assert:
      - equals: "yes"
  - id: flaky
    prompt: "say yes"
    repeat: 5
    pass_rate: 0.6
    assert:
      - equals: "yes"
  - id: too-flaky
    prompt: "say yes"
    repeat: 4
    pass_rate: 0.75
    assert:
      - equals: "yes"
"""

OUTPUTS = {
    "good": "yes",
    "bad": "no",
    "flaky": ["yes", "no", "yes", "yes", "no"],
    "too-flaky": ["yes", "no", "yes", "no"],
}


def test_statuses_and_pass_rates(make_suite):
    run = run_suite(make_suite(SUITE), FakeProvider(OUTPUTS))
    by_id = {c.case.id: c for c in run.cases}
    assert [c.case.id for c in run.cases] == ["good", "bad", "flaky", "too-flaky"]
    assert by_id["good"].status == PASS
    assert by_id["bad"].status == FAIL
    assert by_id["bad"].headline() == 'equals "yes": got "no"'
    flaky = by_id["flaky"]
    assert (flaky.passed_samples, len(flaky.samples), flaky.status) == (3, 5, PASS)
    assert flaky.flaky
    assert flaky.headline() == ""
    too_flaky = by_id["too-flaky"]
    assert (too_flaky.pass_rate, too_flaky.status) == (0.5, FAIL)
    assert (run.passed, run.failed, run.errored, run.total, run.flaky) == (2, 2, 0, 4, 2)
    assert run.sample_total == 11
    assert run.pass_rate == 0.5
    assert not run.ok()
    assert run.ok(0.5)
    assert not run.ok(0.51)


def test_requests_carry_case_settings(make_suite):
    provider = FakeProvider(OUTPUTS)
    run_suite(make_suite(SUITE), provider)
    request = next(r for r in provider.requests if r.case_id == "good")
    assert request.model == "test-model"
    assert request.timeout == 12
    assert request.messages == ({"role": "user", "content": "say yes"},)
    assert sorted(r.sample for r in provider.requests if r.case_id == "flaky") == [0, 1, 2, 3, 4]


def test_overrides_for_model_timeout_and_repeat(make_suite):
    provider = FakeProvider(OUTPUTS)
    suite = make_suite(SUITE)
    run = run_suite(suite, provider, model="other", timeout=3, repeat=2)
    assert {r.model for r in provider.requests} == {"other"}
    assert {r.timeout for r in provider.requests} == {3}
    assert all(len(c.samples) == 2 for c in run.cases)
    assert suite.cases[2].repeat == 5, "the suite itself must not be mutated"


def test_errors_never_abort_the_run(make_suite):
    provider = FakeProvider(
        OUTPUTS,
        raise_for={"bad": ProviderError("HTTP 500: overloaded"), "flaky": RuntimeError("boom")},
    )
    run = run_suite(make_suite(SUITE), provider)
    by_id = {c.case.id: c for c in run.cases}
    assert by_id["good"].status == PASS
    assert by_id["bad"].status == ERROR
    assert by_id["bad"].headline() == "error: HTTP 500: overloaded"
    assert by_id["flaky"].status == ERROR
    assert by_id["flaky"].samples[0].error == "internal error: RuntimeError: boom"
    assert by_id["flaky"].samples[0].internal
    assert not by_id["bad"].samples[0].internal, "provider errors are case errors, not tool bugs"
    assert {case_id for case_id, _, _ in run.internal_errors} == {"flaky"}
    assert by_id["too-flaky"].status == FAIL
    assert run.errored == 2


def test_partial_errors_count_as_failed_samples(make_suite):
    class SometimesBroken(Provider):
        name = "sometimes"

        def complete(self, request: CompletionRequest) -> Completion:
            if request.sample == 1:
                raise ProviderError("timed out after 12s")
            return Completion(text="yes")

    suite = make_suite(SUITE)
    run = run_suite(suite, SometimesBroken(), cases=[suite.cases[2]])
    (case,) = run.cases
    assert [s.status for s in case.samples] == [PASS, ERROR, PASS, PASS, PASS]
    assert case.status == PASS  # 4/5 >= 60%


def test_refusal_flag_reaches_assertions(make_suite):
    suite = make_suite(
        """
        name: Refusals
        cases:
          - {id: polite, prompt: hi, assert: [not_refusal]}
        """
    )
    run = run_suite(suite, FakeProvider({"polite": Completion(text="", refused=True)}))
    assert run.cases[0].status == FAIL
    assert run.cases[0].samples[0].refused


class SlowProvider(Provider):
    name = "slow"

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.peak = 0
        self.calls = 0
        self.lock = threading.Lock()

    def complete(self, request: CompletionRequest) -> Completion:
        with self.lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.03)
        with self.lock:
            self.active -= 1
        return Completion(text="yes")


def test_concurrency_is_bounded(make_suite):
    suite = make_suite(SUITE)
    parallel = SlowProvider()
    run_suite(suite, parallel, concurrency=4)
    assert 1 < parallel.peak <= 4
    serial = SlowProvider()
    run_suite(suite, serial, concurrency=1)
    assert serial.peak == 1


def test_progress_callback(make_suite):
    seen = []
    run_suite(make_suite(SUITE), FakeProvider(OUTPUTS), on_sample=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (11, 11)
    assert [d for d, _ in seen] == list(range(1, 12))


# ----------------------------------------------------------------- regression tests (QA)


def test_interrupt_cancels_queued_requests(make_suite):
    # M6: Ctrl-C used to let every queued request run (12/12 sent, 13.6 s to exit).
    suite = make_suite(
        "name: Many\ndefaults: {model: m}\ncases:\n"
        + "".join(f"  - {{id: c{i}, prompt: hi, assert: [{{min_words: 1}}]}}\n" for i in range(20))
    )
    provider = SlowProvider()

    def interrupt(done: int, total: int) -> None:
        raise KeyboardInterrupt

    started = time.monotonic()
    with pytest.raises(RunInterrupted) as info:
        run_suite(suite, provider, concurrency=2, on_sample=interrupt)
    time.sleep(0.2)  # let any in-flight request finish
    assert info.value.total == 20
    assert info.value.completed == 1
    assert provider.calls <= 4, f"queued requests must be cancelled, {provider.calls} were sent"
    assert time.monotonic() - started < 1


def test_case_with_no_passing_sample_never_passes(make_suite):
    suite = make_suite("name: E\ncases:\n  - {id: e, prompt: hi, assert: [json_valid]}\n")
    lenient = dataclasses.replace(suite.cases[0], pass_rate=0.0)  # bypassing validation on purpose
    run = run_suite(suite, FakeProvider({}, raise_for={"e": ProviderError("HTTP 500")}), cases=[lenient])
    assert run.cases[0].status == ERROR
    assert not run.cases[0].passed
    assert not run.ok(0.0001)
