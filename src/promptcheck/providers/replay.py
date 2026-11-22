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


def load_recordings(path: str | Path) -> dict[str, list[RecordedOutput]]:
    """Load a recording file in either the full or the shorthand format."""
    file = Path(path)
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{file}: replay file not found (create it with --record)") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{file}: not valid JSON ({exc.msg} at line {exc.lineno})") from None
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
    recordings: dict[str, list[RecordedOutput]] = {}
    for case_id, raw in cases.items():
        entries = raw if isinstance(raw, list) else [raw]
        recordings[str(case_id)] = [
            _parse_entry(file, str(case_id), i, entry) for i, entry in enumerate(entries)
        ]
    return recordings


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
        existing: dict[str, Any] = {}
        if file.exists():
            try:
                current = json.loads(file.read_text(encoding="utf-8"))
                if isinstance(current, dict):
                    cases = current.get("cases") if "cases" in current else current
                    if isinstance(cases, dict):
                        existing = dict(cases)
            except (OSError, json.JSONDecodeError):
                existing = {}
        recorded, skipped = [], []
        with self._lock:
            for case_id, count in expected.items():
                samples = self._entries.get(case_id, {})
                if len(samples) == count and all(i in samples for i in range(count)):
                    existing[case_id] = [samples[i] for i in range(count)]
                    recorded.append(case_id)
                else:
                    skipped.append(case_id)
        document = {
            "format": FORMAT,
            "suite": suite_name,
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "recorded_with": f"promptcheck {__version__}",
            "provider": self.inner.label,
            "cases": existing,
        }
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return recorded, skipped
