from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pydantic

from curry_leaves.core.tools import Risk, ToolResult
from curry_leaves.tools.glob import glob_to_regexp, walk

if TYPE_CHECKING:
    import asyncio

    from curry_leaves.providers.base import Context

_MAX_FILE_BYTES = 2_000_000


class _Args(pydantic.BaseModel):
    pattern: str = pydantic.Field(description="Regular expression to match against each line.")
    path: str = pydantic.Field(default=".", description="Directory to search under (default: cwd).")
    glob: str = pydantic.Field(default="**/*", description="Which files to scan, e.g. '**/*.py'.")
    ignore_case: bool = pydantic.Field(default=False, description="Case-insensitive match.")
    max_results: int = pydantic.Field(default=200, description="Cap on matching lines returned.")


def _escape_regexp(s: str) -> str:
    return re.sub(r"[.*+?^${}()|\[\]\\]", lambda m: "\\" + m.group(0), s)


class SearchTool:
    """Search file CONTENTS by regular expression, `find` for filenames."""

    name = "search"
    description = (
        "Search file CONTENTS by regular expression and return matching lines as `path:line: text` "
        "(read-only — works in plan mode). Use `find` for filenames, `search` for what's inside "
        "them. Restrict with `path` (dir) and `glob` (e.g. '*.py')."
    )
    schema: type[pydantic.BaseModel] = _Args
    risk: Risk | None = "read"
    timeout: float | None = None

    async def run(self, args: _Args, ctx: "Context", signal: "asyncio.Event") -> ToolResult:
        try:
            if not Path(args.path).is_dir():
                return ToolResult(content=f"Not a directory: {args.path}", is_error=True)
        except OSError:
            return ToolResult(content=f"Not a directory: {args.path}", is_error=True)

        flags = re.IGNORECASE if args.ignore_case else 0
        note = ""
        try:
            regex = re.compile(args.pattern, flags)
        except re.error:
            regex = re.compile(_escape_regexp(args.pattern), flags)  # fall back to literal match
            note = f"(invalid regex — matching '{args.pattern}' literally)\n"

        file_glob = glob_to_regexp(args.glob)
        hits: list[str] = []
        scanned = 0
        truncated = False

        async for entry in walk(args.path):
            if entry.is_dir or not file_glob.match(entry.rel):
                continue
            full = os.path.join(args.path, entry.rel)
            try:
                info = os.stat(full)
            except OSError:
                continue
            if info.st_size > _MAX_FILE_BYTES:
                continue
            try:
                text = Path(full).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue  # unreadable or binary
            scanned += 1
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if regex.search(line):
                    hits.append(f"{full}:{i + 1}: {line.strip()[:200]}")
                    if len(hits) >= args.max_results:
                        truncated = True
                        break
            if truncated:
                break

        if len(hits) == 0:
            return ToolResult(
                content=(
                    f"No matches for '{args.pattern}' in {args.path} (scanned {scanned} files)."
                    f"{' ' + note.strip() if note else ''}"
                )
            )
        suffix = (
            f"\n... [stopped at {args.max_results} matches; narrow the pattern or glob]"
            if truncated
            else ""
        )
        return ToolResult(content=note + "\n".join(hits) + suffix)


def search_tool() -> SearchTool:
    return SearchTool()
