"""A tiny glob engine shared by ``find`` and ``search``. Supports star (any chars except ``/``),
``?`` (one char except ``/``), and double-star (any depth, collapsing zero or more dirs).
Matches POSIX-style relative paths. Enough for the common patterns; not a full spec.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import AsyncIterator

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".curry-leaves",
    "dist",
    "build",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
}

_SPECIAL_CHARS = re.compile(r"[.+^${}()|\[\]\\]")


def glob_to_regexp(glob: str) -> re.Pattern[str]:
    re_str = ""
    i = 0
    n = len(glob)
    while i < n:
        c = glob[i]
        if c == "*":
            if i + 1 < n and glob[i + 1] == "*":
                i += 1
                if i + 1 < n and glob[i + 1] == "/":
                    i += 1
                    re_str += "(?:.*/)?"
                else:
                    re_str += ".*"
            else:
                re_str += "[^/]*"
        elif c == "?":
            re_str += "[^/]"
        else:
            re_str += _SPECIAL_CHARS.sub(lambda m: "\\" + m.group(0), c)
        i += 1
    return re.compile(f"^{re_str}$")


@dataclass
class WalkEntry:
    """rel: POSIX relative path from the walk root."""

    rel: str
    is_dir: bool


async def walk(base: str, prefix: str = "") -> AsyncIterator[WalkEntry]:
    """Recursively walk `base`, yielding entries (dirs and files), pruning EXCLUDE_DIRS."""
    try:
        entries = list(os.scandir(os.path.join(base, prefix)))
    except OSError:
        return
    for e in entries:
        if e.is_dir(follow_symlinks=False) and e.name in EXCLUDE_DIRS:
            continue
        rel = f"{prefix}/{e.name}" if prefix else e.name
        if e.is_dir(follow_symlinks=False):
            yield WalkEntry(rel=rel, is_dir=True)
            async for sub in walk(base, rel):
                yield sub
        elif e.is_file(follow_symlinks=False):
            yield WalkEntry(rel=rel, is_dir=False)
