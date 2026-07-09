"""Small path helpers — and the one place that decides WHERE curry-leaves keeps its files.

Precedence for the base dir: set_home() > $CURRY_LEAVES_HOME > ~/.curry-leaves.
"""

from __future__ import annotations

import os
from pathlib import Path

# Re-exported for callers that want the same primitives this module uses,
# mirroring the TS `export { isAbsolute, resolve, join }`.
from os.path import isabs as is_absolute
from os.path import join
from os.path import abspath as resolve

__all__ = [
    "set_home",
    "home",
    "sessions_dir",
    "session_dir",
    "session_meta_file",
    "session_transcript_file",
    "find_up",
    "repo_root",
    "find_up_dir",
    "is_absolute",
    "resolve",
    "join",
]

_home_override: str | None = None


def set_home(path: str | None) -> None:
    global _home_override
    _home_override = path


def home() -> str:
    if _home_override is not None:
        return _home_override
    env = os.environ.get("CURRY_LEAVES_HOME")
    return env if env else join(str(Path.home()), ".curry-leaves")


def sessions_dir() -> str:
    """Root directory holding every session's folder: `<home>/sessions`."""
    return join(home(), "sessions")


def session_dir(id: str) -> str:
    """One session's own directory: `<home>/sessions/<id>` (holds meta.json + transcript.jsonl)."""
    return join(sessions_dir(), id)


def session_meta_file(id: str) -> str:
    """Path to a session's static metadata file."""
    return join(session_dir(id), "meta.json")


def session_transcript_file(id: str) -> str:
    """Path to a session's append-only JSONL transcript."""
    return join(session_dir(id), "transcript.jsonl")


def _ancestors(start: str) -> list[str]:
    dirs: list[str] = []
    dir_ = resolve(start)
    while True:
        dirs.append(dir_)
        parent = os.path.dirname(dir_)
        if parent == dir_:
            break
        dir_ = parent
    return dirs


def find_up(relative: str, start: str | None = None) -> str | None:
    """Walk up from `start` (cwd by default) returning the nearest ancestor containing `relative` (a file)."""
    for dir_ in _ancestors(start if start is not None else os.getcwd()):
        candidate = join(dir_, relative)
        if os.path.isfile(candidate):
            return candidate
    return None


def repo_root(start: str | None = None) -> str:
    """The repo root — nearest ancestor containing `.git` (dir or file), else `start` itself."""
    start = start if start is not None else os.getcwd()
    for dir_ in _ancestors(resolve(start)):
        if os.path.exists(join(dir_, ".git")):
            return dir_
    return resolve(start)


def find_up_dir(relative: str, start: str | None = None) -> str | None:
    """Like find_up, but for a directory."""
    for dir_ in _ancestors(start if start is not None else os.getcwd()):
        candidate = join(dir_, relative)
        if os.path.isdir(candidate):
            return candidate
    return None
