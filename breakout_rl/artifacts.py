"""Repository artifact paths and hashes shared by analysis tools."""

from __future__ import annotations

import hashlib
from pathlib import Path


def repository_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a working directory or file path."""

    origin = Path.cwd() if start is None else Path(start)
    current = origin.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "eval" / "breakout_contract_v2.json").is_file():
            return candidate
    raise FileNotFoundError("could not locate the repository root")


def repository_relative_path(path: str | Path, *, root: str | Path) -> str:
    """Return a repository-relative POSIX path or fail for external files."""

    source = Path(path).resolve()
    repository = Path(root).resolve()
    try:
        return source.relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError(
            "artifact metadata only permits repository-relative paths: " f"{path}"
        ) from error


def sha256_file(path: str | Path, *, normalize_text: bool = False) -> str:
    """Hash one artifact, optionally normalizing CRLF text checkout endings."""

    source = Path(path)
    if normalize_text:
        content = source.read_bytes().replace(bytes([13, 10]), bytes([10]))
        return hashlib.sha256(content).hexdigest()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["repository_relative_path", "repository_root", "sha256_file"]
