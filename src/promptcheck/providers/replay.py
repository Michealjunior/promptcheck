"""Replay recorded outputs, and record live ones for later replay.

Recording file format (``promptcheck-recording/1``)::

    {
      "format": "promptcheck-recording/1",
      "suite": "Helpdesk assistant",
      "recorded_at": "2026-11-23T20:15:00Z",
      "cases": {
        "triage-double-charge": [
          {"output": "{...}", "model": "gpt-4.1-mini", "request_hash": "3f9c..."}
        ]
      }
    }

Each case maps to one entry per sample (``repeat``). ``request_hash`` is optional;
when present, replay warns if the prompt, system prompt or model changed since
recording. A hand-written shorthand is also accepted: ``{"case-id": "output"}`` or
``{"case-id": ["sample 1", "sample 2"]}``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from promptcheck import __version__
from promptcheck.errors import ConfigError, ProviderError
from promptcheck.files import FileProblem, atomic_write_text, check_writable, read_utf8
from promptcheck.providers.base import Completion, CompletionRequest, Provider

FORMAT = "promptcheck-recording/1"


@dataclass(frozen=True)
class RecordedOutput:
    output: str
    model: str | None = None
    request_hash: str | None = None
    refused: bool = False


def _parse_entry(path: Path, case_id: str, index: int, raw: Any) -> RecordedOutput:
    where = f"{path}: case '{case_id}', sample #{index + 1}"
    if isinstance(raw, str):
        return RecordedOutput(output=raw)
    if not isinstance(raw, dict) or not isinstance(raw.get("output"), str):
        raise ConfigError(f"{where}: each recording must be a string or an object with 'output'")
    model = raw.get("model")
    request_hash = raw.get("request_hash")
    return RecordedOutput(
        output=raw["output"],
        model=model if isinstance(model, str) else None,
        request_hash=request_hash if isinstance(request_hash, str) else None,
        refused=raw.get("refused") is True,
    )


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


class _DuplicateJSONKey(ValueError):
    pass


def read_recording_cases(path: str | Path) -> dict[str, Any]:
    """Read a recording file and return its raw ``cases`` mapping, validating every entry."""
    file = Path(path)
    try:
        text = read_utf8(file, "recording file")
    except FileProblem as exc:
        message = str(exc)
        if message.endswith("recording file not found"):
            message += " (create it with --record)"
        raise ConfigError(message) from None
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicates)
    except _DuplicateJSONKey as exc:
        raise ConfigError(f"{file}: duplicate key {str(exc)!r}; each case may appear only once") from None
    except json.JSONDecodeError as exc:
        hint = "; it contains merge-conflict markers" if "<<<<<<<" in text or ">>>>>>>" in text else ""
        raise ConfigError(f"{file}: not valid JSON ({exc.msg} at line {exc.lineno}){hint}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{file}: expected a JSON object mapping case ids to outputs")
    if "format" in data or "cases" in data:
        fmt = data.get("format")
        if fmt is not None and fmt != FORMAT:
            raise ConfigError(f"{file}: unsupported recording format {fmt!r} (expected {FORMAT})")
        cases = data.get("cases")
        if not isinstance(cases, dict):
            raise ConfigError(f"{file}: 'cases' must map case ids to lists of recordings")
    else:
        cases = data
    for case_id, raw in cases.items():
        entries = raw if isinstance(raw, list) else [raw]
        if not entries:
            raise ConfigError(f"{file}: case '{case_id}' has an empty list of recordings")
        for index, entry in enumerate(entries):
            _parse_entry(file, str(case_id), index, entry)
    return dict(cases)


def load_recordings(path: str | Path) -> dict[str, list[RecordedOutput]]:
    """Load a recording file in either the full or the shorthand format."""
    file = Path(path)
    recordings: dict[str, list[RecordedOutput]] = {}
    for case_id, raw in read_recording_cases(file).items():
        entries = raw if isinstance(raw, list) else [raw]
        recordings[str(case_id)] = [
            _parse_entry(file, str(case_id), i, entry) for i, entry in enumerate(entries)
        ]
    return recordings


def check_recording_target(path: str | Path) -> None:
    """Fail before a run if ``--record`` could not safely update ``path``."""
    file = Path(path)
    try:
        check_writable(file, "recording")
    except FileProblem as exc:
        raise ConfigError(str(exc)) from None
    if file.exists():
        _existing_cases(file)


def _existing_cases(file: Path) -> dict[str, Any]:
    try:
        return read_recording_cases(file)
    except ConfigError as exc:
        raise ConfigError(
            f"refusing to overwrite {file}: it is not a valid recording ({exc}). "
            "Fix it (for example resolve merge conflicts) or delete it, then run again"
        ) from None


class ReplayProvider(Provider):
    """Serves recorded outputs: deterministic, offline and free."""

    name = "replay"

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.recordings = load_recordings(self.path)

    @property
    def label(self) -> str:
        return f"replay ({self.path.as_posix()})"

    def complete(self, request: CompletionRequest) -> Completion:
        entries = self.recordings.get(request.case_id)
        if not entries:
            raise ProviderError(
                f"no recording for case '{request.case_id}' in {self.path}; "
                "record it with a live provider and --record"
            )
        if request.sample >= len(entries):
            raise ProviderError(
                f"case '{request.case_id}' has {len(entries)} recorded output(s) but sample "
                f"#{request.sample + 1} was requested; lower 'repeat' or re-record"
            )
        entry = entries[request.sample]
        warnings = []
        if entry.request_hash and entry.request_hash != request.fingerprint():
            warnings.append(
                "stale recording: the prompt, system prompt or model changed after this output "
                "was recorded; re-record with --record"
            )
        return Completion(
            text=entry.output,
            model=entry.model,
            refused=entry.refused,
            finish_reason="replay",
            warnings=warnings,
        )


class RecordingProvider(Provider):
    """Wraps a live provider and keeps every successful output for ``save``."""

    def __init__(self, inner: Provider) -> None:
        super().__init__()
        self.inner = inner
        self.name = inner.name
        self._lock = threading.Lock()
        self._entries: dict[str, dict[int, dict[str, Any]]] = {}

    @property
    def label(self) -> str:
        return self.inner.label

    @property
    def notices(self) -> list[str]:
        return self.inner.notices

    def complete(self, request: CompletionRequest) -> Completion:
        completion = self.inner.complete(request)
        entry: dict[str, Any] = {
            "output": completion.text,
            "model": completion.model or request.model,
            "request_hash": request.fingerprint(),
        }
        if completion.refused:
            entry["refused"] = True
        with self._lock:
            self._entries.setdefault(request.case_id, {})[request.sample] = entry
        return completion

    def save(
        self, path: str | Path, *, suite_name: str, expected: dict[str, int]
    ) -> tuple[list[str], list[str]]:
        """Merge recordings into ``path``.

        ``expected`` maps case id to its sample count. A case is written only if every
        sample succeeded. Returns ``(recorded_ids, skipped_ids)``.
        """
        file = Path(path)
        recorded, skipped = [], []
        new_cases: dict[str, Any] = {}
        with self._lock:
            for case_id, count in expected.items():
                samples = self._entries.get(case_id, {})
                if len(samples) == count and all(i in samples for i in range(count)):
                    new_cases[case_id] = [samples[i] for i in range(count)]
                    recorded.append(case_id)
                else:
                    skipped.append(case_id)
        try:
            existing = _existing_cases(file) if file.exists() else {}
        except ConfigError as exc:
            # The file became unreadable during the run: keep it, and keep the new outputs too.
            fallback = file.with_name(f"{file.stem}.new{file.suffix or '.json'}")
            atomic_write_text(fallback, self._document(suite_name, new_cases))
            raise ConfigError(f"{exc}. The new outputs were saved to {fallback} instead") from None
        existing.update(new_cases)
        atomic_write_text(file, self._document(suite_name, existing))
        return recorded, skipped

    def _document(self, suite_name: str, cases: dict[str, Any]) -> str:
        document = {
            "format": FORMAT,
            "suite": suite_name,
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "recorded_with": f"promptcheck {__version__}",
            "provider": self.inner.label,
            "cases": cases,
        }
        return json.dumps(document, indent=2, ensure_ascii=False) + "\n"
