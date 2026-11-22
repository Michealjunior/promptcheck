"""Run a suite against a provider and collect results."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from promptcheck.assertions import AssertionResult, evaluate_all
from promptcheck.errors import PromptcheckError
from promptcheck.providers.base import CompletionRequest, Provider
from promptcheck.suite import Case, Suite

PASS, FAIL, ERROR = "pass", "fail", "error"


@dataclass
class SampleResult:
    """One call to the model for one case, and how its assertions went."""

    index: int
    output: str | None = None
    assertions: list[AssertionResult] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    refused: bool = False
    model: str | None = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return ERROR
        return PASS if all(a.passed for a in self.assertions) else FAIL

    @property
    def failures(self) -> list[AssertionResult]:
        return [a for a in self.assertions if not a.passed]


@dataclass
class CaseResult:
    case: Case
    samples: list[SampleResult]

    @property
    def passed_samples(self) -> int:
        return sum(1 for s in self.samples if s.status == PASS)

    @property
    def pass_rate(self) -> float:
        return self.passed_samples / len(self.samples) if self.samples else 0.0

    @property
    def threshold(self) -> float:
        return self.case.pass_rate

    @property
    def passed(self) -> bool:
        return bool(self.samples) and self.pass_rate + 1e-9 >= self.threshold

    @property
    def status(self) -> str:
        if self.passed:
            return PASS
        if all(s.status == ERROR for s in self.samples):
            return ERROR
        return FAIL

    @property
    def flaky(self) -> bool:
        """Some samples passed and some did not."""
        return 0 < self.passed_samples < len(self.samples)

    @property
    def latency_ms(self) -> float:
        timed = [s.latency_ms for s in self.samples if s.error is None]
        return sum(timed) / len(timed) if timed else 0.0

    @property
    def warnings(self) -> list[str]:
        seen: list[str] = []
        for sample in self.samples:
            for warning in sample.warnings:
                if warning not in seen:
                    seen.append(warning)
        return seen

    def headline(self) -> str:
        """One-line reason for a failure (empty for passing cases)."""
        if self.passed:
            return ""
        for sample in self.samples:
            if sample.error:
                return f"error: {sample.error}"
            if sample.failures:
                first = sample.failures[0]
                return f"{first.label}: {first.detail}" if first.detail else first.label
        return ""


@dataclass
class RunResult:
    suite: Suite
    provider: str
    model: str | None
    cases: list[CaseResult]
    started_at: datetime
    duration_s: float
    notices: list[str] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for c in self.cases if c.status == status)

    @property
    def passed(self) -> int:
        return self.count(PASS)

    @property
    def failed(self) -> int:
        return self.count(FAIL)

    @property
    def errored(self) -> int:
        return self.count(ERROR)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def flaky(self) -> int:
        return sum(1 for c in self.cases if c.flaky)

    @property
    def sample_total(self) -> int:
        return sum(len(c.samples) for c in self.cases)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def ok(self, min_pass_rate: float = 1.0) -> bool:
        return self.total > 0 and self.pass_rate + 1e-9 >= min_pass_rate


def build_request(case: Case, sample: int, default_model: str | None, timeout: float) -> CompletionRequest:
    return CompletionRequest(
        case_id=case.id,
        sample=sample,
        model=case.model or default_model,
        system=case.system,
        messages=tuple(dict(m) for m in case.messages),
        temperature=case.temperature,
        max_tokens=case.max_tokens,
        timeout=timeout,
    )


def run_sample(
    provider: Provider, case: Case, sample: int, default_model: str | None, timeout: float
) -> SampleResult:
    """Call the provider once and grade the output. Never raises."""
    started = time.perf_counter()
    try:
        request = build_request(case, sample, default_model, timeout)
        completion = provider.complete(request)
    except PromptcheckError as exc:
        return SampleResult(index=sample, error=str(exc), latency_ms=_ms(started))
    except Exception as exc:  # a bug or unexpected failure must not abort the run
        return SampleResult(index=sample, error=f"{type(exc).__name__}: {exc}", latency_ms=_ms(started))
    latency = _ms(started)
    return SampleResult(
        index=sample,
        output=completion.text,
        assertions=evaluate_all(case.checks, completion.text, completion.refused),
        latency_ms=latency,
        warnings=list(completion.warnings),
        refused=completion.refused,
        model=completion.model,
    )


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def run_suite(
    suite: Suite,
    provider: Provider,
    *,
    model: str | None = None,
    cases: list[Case] | None = None,
    concurrency: int = 4,
    timeout: float | None = None,
    repeat: int | None = None,
    on_sample: Callable[[int, int], None] | None = None,
) -> RunResult:
    """Run every (case, sample) pair on a thread pool and return ordered results.

    ``on_sample(done, total)`` is called after each sample for progress display.
    """
    selected = list(suite.cases if cases is None else cases)
    if repeat is not None:
        selected = [replace(case, repeat=repeat) for case in selected]
    default_model = model or suite.provider.model
    request_timeout = timeout if timeout is not None else suite.timeout
    jobs = [(case, i) for case in selected for i in range(case.repeat)]
    results: dict[tuple[str, int], SampleResult] = {}
    started_at = datetime.now(timezone.utc)
    clock = time.perf_counter()
    workers = max(1, min(concurrency, len(jobs) or 1))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="promptcheck") as pool:
        futures = {
            pool.submit(run_sample, provider, case, i, default_model, request_timeout): (case.id, i)
            for case, i in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if on_sample:
                on_sample(done, len(jobs))
    case_results = [
        CaseResult(case=case, samples=[results[(case.id, i)] for i in range(case.repeat)])
        for case in selected
    ]
    return RunResult(
        suite=suite,
        provider=provider.label,
        model=default_model,
        cases=case_results,
        started_at=started_at,
        duration_s=time.perf_counter() - clock,
        notices=provider.notices,
    )
