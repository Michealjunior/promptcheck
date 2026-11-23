"""Small file helpers: strict UTF-8 reading, writability checks and atomic writes."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


class FileProblem(Exception):
    """A file cannot be read or written; the message is ready to show to a user."""


def read_utf8(path: str | Path, what: str = "file") -> str:
    """Read ``path`` as UTF-8 text (a UTF-8 BOM is allowed) with readable errors."""
    file = Path(path)
    try:
        data = file.read_bytes()
    except FileNotFoundError:
        raise FileProblem(f"{file}: {what} not found") from None
    except IsADirectoryError:
        raise FileProblem(f"{file}: is a directory, expected a {what}") from None
    except PermissionError:
        raise FileProblem(f"{file}: permission denied") from None
    except OSError as exc:
        raise FileProblem(f"{file}: cannot read {what} ({exc.strerror or exc})") from None
    if data.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in data[:512]:
        raise FileProblem(f"{file}: looks like UTF-16 or UTF-32 text; save the {what} as UTF-8")
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = data[: exc.start].count(b"\n") + 1
        raise FileProblem(
            f"{file}: is not valid UTF-8 (byte 0x{data[exc.start]:02x} on line {line}); "
            f"save the {what} as UTF-8"
        ) from None


def check_writable(path: str | Path, what: str = "file") -> None:
    """Raise :class:`FileProblem` unless ``path`` can be created or overwritten.

    Missing parent directories are fine as long as they can be created inside an
    existing, writable directory.
    """
    target = Path(path)
    if target.exists():
        if target.is_dir():
            raise FileProblem(f"{target}: is a directory, cannot write the {what} there")
        if not os.access(target, os.W_OK):
            raise FileProblem(f"{target}: permission denied, cannot write the {what}")
        return
    parent = target.parent
    while not parent.exists():
        if parent == parent.parent:
            break
        parent = parent.parent
    if not parent.is_dir():
        raise FileProblem(f"{target}: {parent} is not a directory, cannot write the {what}")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise FileProblem(f"{target}: permission denied in {parent}, cannot write the {what}")


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` via a temporary file and rename, so readers never see half a file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        if target.exists():
            with contextlib.suppress(OSError):
                os.chmod(tmp_name, target.stat().st_mode & 0o777)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return target
