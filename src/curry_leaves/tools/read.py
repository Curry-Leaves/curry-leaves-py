"""The `read` tool: read a local file, or a resource URL (artifact:// / local:// / skill://)."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import pydantic

from curry_leaves.core.tools import Risk, Tool, ToolResult
from curry_leaves.providers.base import Context
from curry_leaves.util.resources import Resolvers, is_resource_url, resolve_url

MAX_LINES = 2000


class ReadArgs(pydantic.BaseModel):
    path: str = pydantic.Field(description="File path or resource URL (artifact://… / local://…).")
    offset: Optional[int] = pydantic.Field(default=None, description="1-based line to start from.")
    limit: Optional[int] = pydantic.Field(default=None, description="Maximum number of lines to return.")


class ReadTool:
    """Structurally satisfies the `Tool` protocol (see core/tools.py)."""

    name = "read"
    description = (
        "Read a UTF-8 text file and return its contents with line numbers. Also reads internal "
        "resources by URL: artifact://<id> (full output a tool truncated) and local://<slug> "
        "(a session resource, e.g. a plan). Optionally pass `offset` (1-based start line) and "
        "`limit` (max lines)."
    )
    schema: type[pydantic.BaseModel] = ReadArgs
    risk: Optional[Risk] = "read"
    timeout: Optional[float] = None

    async def run(self, args: pydantic.BaseModel, ctx: Context, signal: asyncio.Event) -> ToolResult:
        assert isinstance(args, ReadArgs)
        if is_resource_url(args.path):
            try:
                text = resolve_url(
                    args.path,
                    Resolvers(
                        blobs=ctx.blobs,
                        resolve_local=ctx.resolve_local,
                        resolve_skill=ctx.resolve_skill,
                    ),
                )
            except Exception as e:
                return ToolResult(content=f"Could not read {args.path}: {e}", is_error=True)
        else:
            try:
                with open(args.path, encoding="utf-8") as f:
                    text = f.read()
            except FileNotFoundError:
                return ToolResult(content=f"File not found: {args.path}", is_error=True)
            except IsADirectoryError:
                return ToolResult(content=f"Is a directory, not a file: {args.path}", is_error=True)
            except OSError as e:
                return ToolResult(content=f"Could not read {args.path}: {e}", is_error=True)

        lines = text.split("\n")
        start = max(0, (args.offset if args.offset is not None else 1) - 1)
        end = start + args.limit if args.limit is not None else len(lines)
        selected = lines[start:end]

        truncated = False
        if len(selected) > MAX_LINES:
            selected = selected[:MAX_LINES]
            truncated = True

        if len(selected) == 0:
            return ToolResult(content="(no lines in range)" if lines else "(empty file)")

        body = "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(selected))
        suffix = "\n... [truncated to 2000 lines; pass offset/limit for more]" if truncated else ""
        return ToolResult(content=body + suffix)


def read_tool() -> Tool[Any]:
    return ReadTool()
