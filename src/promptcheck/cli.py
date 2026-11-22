"""Command-line interface: ``promptcheck run`` and ``promptcheck validate``."""

from __future__ import annotations

import argparse
import fnmatch
import sys
from collections.abc import Sequence
from typing import TextIO

from promptcheck import __version__
from promptcheck.errors import ConfigError, PromptcheckError, SuiteError
from promptcheck.providers import RecordingProvider, RetryPolicy, create_provider
from promptcheck.reporters import TerminalReporter, write_html, write_junit, write_markdown
from promptcheck.runner import run_suite
from promptcheck.suite import PROVIDERS, load_suite, parse_rate

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


def _rate(value: str) -> float:
    try:
        return parse_rate(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"invalid rate {value!r}: use a fraction such as 0.9 or a percentage such as 90%"
        ) from None


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive whole number, got {value!r}")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = -1
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected zero or a positive whole number, got {value!r}")
    return number


def _seconds(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        number = 0
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number of seconds, got {value!r}")
    return number


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
    source.add_argument("--model", help="override the suite's default model")
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
        "--repeat", type=_positive_int, metavar="N", help="run every case N times (overrides the suite)"
    )
    control.add_argument(
        "--concurrency",
        "-j",
        type=_positive_int,
        default=4,
        metavar="N",
        help="parallel requests (default: 4)",
    )
    control.add_argument(
        "--timeout",
        type=_seconds,
        metavar="SECONDS",
        help="per-request timeout (default: the suite's, or 60)",
    )
    control.add_argument(
        "--retries",
        type=_non_negative_int,
        default=3,
        metavar="N",
        help="retries on 429/5xx/network errors with exponential backoff (default: 3)",
    )
    control.add_argument(
        "--min-pass-rate",
        type=_rate,
        default=1.0,
        metavar="RATE",
        help="fraction of cases that must pass for exit code 0 (default: 1.0)",
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
    run = run_suite(
        suite,
        active,
        model=args.model,
        cases=cases,
        concurrency=args.concurrency,
        timeout=args.timeout,
        repeat=args.repeat,
        on_sample=None if args.quiet else _progress(stderr),
    )

    written: dict[str, str] = {}
    if args.junit:
        written["junit"] = str(write_junit(run, args.junit))
    if args.markdown:
        written["markdown"] = str(write_markdown(run, args.markdown, args.min_pass_rate))
    if args.html:
        written["html"] = str(write_html(run, args.html, args.min_pass_rate))
    if recorder is not None:
        expected = {c.case.id: len(c.samples) for c in run.cases}
        recorded, skipped = recorder.save(args.record, suite_name=suite.name, expected=expected)
        written["recording"] = f"{args.record} ({len(recorded)} cases)"
        if skipped:
            stderr.write("promptcheck: not recorded because a sample errored: " + ", ".join(skipped) + "\n")
    reporter.report(run, written, args.min_pass_rate)
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
        stderr.write("promptcheck: interrupted\n")
        return 130
