"""The `find` tool: list files/directories matching a glob pattern. Read-only, so it
works even in plan mode.
"""

from __future__ import annotations

import asyncio
import os
import stat as stat_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic

from curry_leaves.core.tools import Risk, Tool, ToolResult
from curry_leaves.tools.glob import glob_to_regexp, walk

if TYPE_CHECKING:
    from curry_leaves.providers.base import Context


class FindArgs(pydantic.BaseModel):
    path: str = pydantic.Field(default=".", description="Directory to search under (default: cwd).")
    pattern: str = pydantic.Field(default="*", description="Glob pattern, e.g. '*' or '**/*.py'.")
    max_results: int = pydantic.Field(default=200, description="Cap on entries returned.")


class FindTool:
    name = "find"
    description = (
        "List files and directories matching a glob pattern under a path (read-only — works in "
        "plan mode). Use to explore project structure. Patterns: '*' (immediate children), "
        "'**/*.py' (all Python files recursively)."
    )
    schema: type[pydantic.BaseModel] = FindArgs
    risk: Risk | None = "read"
    timeout: float | None = None

    async def run(self, args: pydantic.BaseModel, ctx: "Context", signal: asyncio.Event) -> ToolResult:
        assert isinstance(args, FindArgs)
        try:
            st = os.stat(args.path)
        except OSError:
            return ToolResult(content=f"No such path: {args.path}", is_error=True)
        if not stat_module.S_ISDIR(st.st_mode):
            return ToolResult(content=f"Not a directory: {args.path}", is_error=True)

        re = glob_to_regexp(args.pattern)
        results: list[str] = []
        capped = False
        try:
            async for entry in walk(args.path):
                if not re.match(entry.rel):
                    continue
                full = str(Path(args.path) / entry.rel)
                results.append(f"{full}/" if entry.is_dir else full)
                if len(results) >= args.max_results:
                    capped = True
                    break
        except OSError as e:
            return ToolResult(content=f"find failed: {e}", is_error=True)

        if len(results) == 0:
            return ToolResult(content=f"(no matches for '{args.pattern}' under {args.path})")
        results.sort()
        suffix = f"\n... [capped at {args.max_results}; narrow the pattern]" if capped else ""
        return ToolResult(content="\n".join(results) + suffix)


def find_tool() -> Tool[Any]:
    return FindTool()
