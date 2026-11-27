"""Command-line interface: ``promptcheck run`` and ``promptcheck validate``."""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from promptcheck import __version__
from promptcheck.errors import ConfigError, PromptcheckError, SuiteError
from promptcheck.files import FileProblem, check_writable
from promptcheck.providers import RecordingProvider, RetryPolicy, create_provider
from promptcheck.providers.http import MAX_RETRIES
from promptcheck.providers.replay import check_recording_target
from promptcheck.reporters import TerminalReporter, write_html, write_junit, write_markdown
from promptcheck.runner import RunInterrupted, RunResult, run_suite
from promptcheck.suite import MAX_REPEAT, PROVIDERS, load_suite, parse_rate
from promptcheck.urls import validate_base_url

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130
MAX_CONCURRENCY = 64
MAX_TIMEOUT = 3600.0
ISSUES_URL = "https://github.com/StarJnrDev/promptcheck/issues"


def _rate(value: str) -> float:
    try:
        return parse_rate(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"invalid rate {value!r}: use a fraction above 0 and at most 1, such as 0.9 or 90%"
        ) from None


def _int_between(low: int, high: int):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            number = low - 1
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"expected a whole number from {low} to {high}, got {value!r}")
        return number

    return parse


def _seconds(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        number = 0
    if not 0 < number <= MAX_TIMEOUT:  # also rejects nan
        raise argparse.ArgumentTypeError(
            f"expected a number of seconds above 0 and at most {MAX_TIMEOUT:g}, got {value!r}"
        )
    return number


def _non_empty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptcheck",
        description="Regression tests for LLM prompts. Catch silent breakage before your users do.",
    )
    parser.add_argument("--version", action="version", version=f"promptcheck {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    run = sub.add_parser("run", help="run a suite and report the results")
    run.add_argument("suite", help="path to a YAML suite file")
    source = run.add_argument_group("provider")
    source.add_argument("--provider", choices=PROVIDERS, help="override the suite's provider")
    source.add_argument("--model", type=_non_empty, help="override the suite's default model")
    source.add_argument("--base-url", help="override the API base URL (OpenAI-compatible or Anthropic)")
    source.add_argument("--replay", metavar="PATH", help="serve recorded outputs from PATH (offline)")
    source.add_argument("--record", metavar="PATH", help="save live outputs to PATH for later replay")
    reports = run.add_argument_group("reports")
    reports.add_argument("--junit", metavar="PATH", help="write a JUnit XML report")
    reports.add_argument("--markdown", metavar="PATH", help="write a Markdown summary (for PR comments)")
    reports.add_argument("--html", metavar="PATH", help="write a single-file HTML report")
    control = run.add_argument_group("execution")
    control.add_argument(
        "--case",
        action="append",
        metavar="PATTERN",
        default=[],
        help="only run cases whose id matches PATTERN (glob, repeatable)",
    )
    control.add_argument(
        "--repeat",
        type=_int_between(1, MAX_REPEAT),
        metavar="N",
        help=f"run every case N times, 1 to {MAX_REPEAT} (overrides the suite)",
    )
    control.add_argument(
        "--concurrency",
        "-j",
        type=_int_between(1, MAX_CONCURRENCY),
        default=4,
        metavar="N",
        help=f"parallel requests, 1 to {MAX_CONCURRENCY} (default: 4)",
    )
    control.add_argument(
        "--timeout",
        type=_seconds,
        metavar="SECONDS",
        help="deadline for each request attempt, including the response body (default: suite's, or 60)",
    )
    control.add_argument(
        "--retries",
        type=_int_between(0, MAX_RETRIES),
        default=3,
        metavar="N",
        help=f"retries on 408/429/5xx/network errors with backoff, 0 to {MAX_RETRIES} (default: 3)",
    )
    control.add_argument(
        "--min-pass-rate",
        type=_rate,
        default=1.0,
        metavar="RATE",
        help="fraction of cases (above 0, at most 1) that must pass for exit code 0 (default: 1.0)",
    )
    output = run.add_argument_group("output")
    output.add_argument("--no-color", action="store_true", help="disable coloured output")
    output.add_argument("-v", "--verbose", action="store_true", help="show every assertion and output")
    output.add_argument("-q", "--quiet", action="store_true", help="print only the summary line")

    validate = sub.add_parser("validate", help="check a suite file without calling a model")
    validate.add_argument("suite", help="path to a YAML suite file")
    return parser


def _progress(stream: TextIO):
    if not getattr(stream, "isatty", lambda: False)():
        return None

    def update(done: int, total: int) -> None:
        if done == total:
            stream.write("\r\033[2K")
        else:
            stream.write(f"\r  running… {done}/{total} samples")
        stream.flush()

    return update


def _check_outputs(args: argparse.Namespace, suite_path: Path) -> dict[str, str]:
    """Fail before any request is sent if a report or recording could not be written."""
    outputs = {
        label: path
        for label, path in (
            ("junit", args.junit),
            ("markdown", args.markdown),
            ("html", args.html),
            ("recording", args.record),
        )
        if path
    }
    seen: dict[Path, str] = {suite_path.resolve(): "suite file"}
    if args.replay:
        seen.setdefault(Path(args.replay).resolve(), "replay file")
    for label, path in outputs.items():
        resolved = Path(path).resolve()
        if resolved in seen:
            raise ConfigError(f"the {label} output {path} is the same file as the {seen[resolved]}")
        seen[resolved] = f"{label} output"
        if label == "recording":
            check_recording_target(path)
        else:
            try:
                check_writable(path, f"{label} report")
            except FileProblem as exc:
                raise ConfigError(str(exc)) from None
    return outputs


def _write_reports(
    run: RunResult,
    args: argparse.Namespace,
    recorder: RecordingProvider | None,
    stderr: TextIO,
) -> tuple[dict[str, str], list[str]]:
    """Write every requested output; collect failures instead of raising."""
    written: dict[str, str] = {}
    problems: list[str] = []
    writers = (
        ("junit", args.junit, lambda path: write_junit(run, path)),
        ("markdown", args.markdown, lambda path: write_markdown(run, path, args.min_pass_rate)),
        ("html", args.html, lambda path: write_html(run, path, args.min_pass_rate)),
    )
    for label, path, write in writers:
        if not path:
            continue
        try:
            written[label] = str(write(path))
        except OSError as exc:
            problems.append(f"could not write the {label} report to {path}: {exc.strerror or exc}")
    if recorder is not None:
        expected = {c.case.id: len(c.samples) for c in run.cases}
        try:
            recorded, skipped = recorder.save(args.record, suite_name=run.suite.name, expected=expected)
            written["recording"] = f"{args.record} ({len(recorded)} cases)"
            if skipped:
                stderr.write(
                    "promptcheck: not recorded because a sample errored: " + ", ".join(skipped) + "\n"
                )
        except ConfigError as exc:
            problems.append(str(exc))
        except OSError as exc:
            fd, rescue = tempfile.mkstemp(prefix="promptcheck-recording-", suffix=".json")
            os.close(fd)
            os.unlink(rescue)  # save() must see a new file, not an empty (invalid) one
            message = f"could not write the recording to {args.record}: {exc.strerror or exc}"
            try:
                recorder.save(rescue, suite_name=run.suite.name, expected=expected)
                message += f"; the outputs were saved to {rescue} instead"
            except (OSError, ConfigError):
                pass
            problems.append(message)
    return written, problems


def command_run(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    suite = load_suite(args.suite)
    cases = suite.cases
    if args.case:
        cases = [c for c in cases if any(fnmatch.fnmatchcase(c.id, p) for p in args.case)]
        if not cases:
            patterns = ", ".join(args.case)
            raise ConfigError(f"no case in {suite.path} matches --case {patterns}")
    if args.record and (args.replay or args.provider == "replay"):
        raise ConfigError("--record needs a live provider; it cannot be combined with replay")
    if args.base_url is not None:
        try:
            validate_base_url(args.base_url)
        except ValueError as exc:
            raise ConfigError(f"--base-url {exc}") from None
    _check_outputs(args, suite.path)

    provider = create_provider(
        suite.provider,
        name=args.provider,
        model=args.model,
        base_url=args.base_url,
        replay=args.replay,
        policy=RetryPolicy(retries=args.retries),
    )
    recorder = RecordingProvider(provider) if args.record else None
    active = recorder or provider

    reporter = TerminalReporter(
        stdout, color=False if args.no_color else None, verbose=args.verbose, quiet=args.quiet
    )
    samples = sum(args.repeat or c.repeat for c in cases)
    reporter.header(suite.name, active.label, args.model or suite.provider.model, len(cases), samples)
    progress = None if args.quiet else _progress(stderr)
    try:
        run = run_suite(
            suite,
            active,
            model=args.model,
            cases=cases,
            concurrency=args.concurrency,
            timeout=args.timeout,
            repeat=args.repeat,
            on_sample=progress,
        )
    except RunInterrupted as exc:
        if progress:
            stderr.write("\r\033[2K")
        reporter.interrupted(exc.completed, exc.total)
        return EXIT_INTERRUPTED

    # Show the results first, so a problem writing files never hides them.
    reporter.report(run, None, args.min_pass_rate)
    written, problems = _write_reports(run, args, recorder, stderr)
    reporter.outputs(written)
    for problem in problems:
        stderr.write(f"promptcheck: error: {problem}\n")
    internal = run.internal_errors
    for case_id, sample, message in internal:
        stderr.write(f"promptcheck: {message} (case '{case_id}', sample {sample + 1})\n")
    if internal:
        stderr.write(
            "promptcheck: this is a bug in promptcheck, not a failure of your prompt; "
            f"please report it at {ISSUES_URL}\n"
        )
    if problems or internal:
        return EXIT_USAGE
    return EXIT_OK if run.ok(args.min_pass_rate) else EXIT_FAILED


def command_validate(args: argparse.Namespace, stdout: TextIO) -> int:
    suite = load_suite(args.suite)
    checks = sum(len(c.checks) for c in suite.cases)
    stdout.write(
        f"{suite.path}: ok · {len(suite.cases)} cases, {suite.sample_count} samples, {checks} assertions\n"
    )
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None, stdout: TextIO | None = None, stderr: TextIO | None = None
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on usage errors and 0 for --help/--version
        return int(exc.code or 0)
    if not args.command:
        parser.print_help(stdout)
        return EXIT_USAGE
    try:
        if args.command == "validate":
            return command_validate(args, stdout)
        return command_run(args, stdout, stderr)
    except (SuiteError, ConfigError) as exc:
        stderr.write(f"promptcheck: error: {exc}\n")
        return EXIT_USAGE
    except PromptcheckError as exc:
        stderr.write(f"promptcheck: error: {exc}\n")
        return EXIT_USAGE
    except KeyboardInterrupt:
        stderr.write("\npromptcheck: interrupted\n")
        return EXIT_INTERRUPTED


def entry() -> None:
    """Console-script entry point."""
    code = main()
    if code == EXIT_INTERRUPTED:
        # Worker threads may still be waiting on in-flight requests; exit without joining them.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)
