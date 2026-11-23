"""JUnit XML, understood by GitHub Actions test reporters, GitLab, Jenkins and most CI UIs."""

from __future__ import annotations

import re
import socket
import xml.etree.ElementTree as ET
from pathlib import Path

from promptcheck import __version__
from promptcheck.files import atomic_write_text
from promptcheck.runner import ERROR, FAIL, CaseResult, RunResult

# XML 1.0 forbids most control characters; model output occasionally contains them.
_INVALID_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def _clean(text: str) -> str:
    return _INVALID_XML.sub("\N{REPLACEMENT CHARACTER}", text)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "suite"


def _case_details(case: CaseResult) -> str:
    lines = [
        f"{case.passed_samples}/{len(case.samples)} samples passed "
        f"(pass rate {case.pass_rate:.0%}, threshold {case.threshold:.0%})"
    ]
    for sample in case.samples:
        prefix = f"sample {sample.index + 1}"
        if sample.error:
            lines.append(f"{prefix}: error: {sample.error}")
            continue
        for result in sample.assertions:
            if not result.passed:
                detail = f": {result.detail}" if result.detail else ""
                lines.append(f"{prefix}: FAILED {result.label}{detail}")
        for warning in sample.warnings:
            lines.append(f"{prefix}: warning: {warning}")
    return "\n".join(lines)


def _outputs(case: CaseResult) -> str:
    blocks = []
    for sample in case.samples:
        if sample.output is None:
            continue
        header = f"--- sample {sample.index + 1} ({sample.status}) ---"
        blocks.append(f"{header}\n{sample.output}")
    return "\n".join(blocks)


def render_junit(run: RunResult) -> str:
    suite_name = run.suite.name
    root = ET.Element(
        "testsuites",
        {
            "name": "promptcheck",
            "tests": str(run.total),
            "failures": str(run.failed),
            "errors": str(run.errored),
            "time": f"{run.duration_s:.3f}",
        },
    )
    testsuite = ET.SubElement(
        root,
        "testsuite",
        {
            "name": suite_name,
            "tests": str(run.total),
            "failures": str(run.failed),
            "errors": str(run.errored),
            "skipped": "0",
            "time": f"{run.duration_s:.3f}",
            "timestamp": run.started_at.strftime("%Y-%m-%dT%H:%M:%S"),
            "hostname": socket.gethostname() or "localhost",
        },
    )
    properties = ET.SubElement(testsuite, "properties")
    for key, value in (
        ("promptcheck.version", __version__),
        ("provider", run.provider),
        ("model", run.model or ""),
        ("suite.file", run.suite.path.as_posix()),
    ):
        ET.SubElement(properties, "property", {"name": key, "value": _clean(value)})
    classname = f"promptcheck.{_slug(suite_name)}"
    for case in run.cases:
        element = ET.SubElement(
            testsuite,
            "testcase",
            {
                "classname": classname,
                "name": case.case.id,
                "time": f"{sum(s.latency_ms for s in case.samples) / 1000:.3f}",
            },
        )
        if case.status == FAIL:
            failure = ET.SubElement(
                element, "failure", {"message": _clean(case.headline()), "type": "AssertionFailed"}
            )
            failure.text = _clean(_case_details(case))
        elif case.status == ERROR:
            error = ET.SubElement(
                element, "error", {"message": _clean(case.headline()), "type": "ProviderError"}
            )
            error.text = _clean(_case_details(case))
        output = _outputs(case)
        if output:
            ET.SubElement(element, "system-out").text = _clean(output)
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def write_junit(run: RunResult, path: str | Path) -> Path:
    return atomic_write_text(path, render_junit(run))
