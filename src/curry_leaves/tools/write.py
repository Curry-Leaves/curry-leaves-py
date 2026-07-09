from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pydantic

from curry_leaves.core.tools import Risk, Tool, ToolResult

if TYPE_CHECKING:
    import asyncio

    from curry_leaves.providers.base import Context


class _Args(pydantic.BaseModel):
    path: str = pydantic.Field(description="File path, relative to the working directory or absolute.")
    content: str = pydantic.Field(description="The COMPLETE contents to write to the file.")


class WriteTool:
    """Satisfies the `Tool` Protocol structurally (see core/tools.py)."""

    name = "write"
    description = (
        "Create a new file (or overwrite an existing one) with the given content. Parent "
        "directories are created automatically. Prefer this over shell redirection (`echo >`, "
        "heredocs) for writing files — pass the COMPLETE file content in one call, not a fragment."
    )
    schema = _Args
    risk: Risk | None = "write"
    timeout: float | None = None

    async def run(self, args: _Args, ctx: "Context", signal: "asyncio.Event") -> ToolResult:
        try:
            parent = os.path.dirname(args.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(args.path, "w", encoding="utf-8") as f:
                f.write(args.content)
        except IsADirectoryError:
            return ToolResult(content=f"Is a directory, not a file: {args.path}", is_error=True)
        except OSError as e:
            return ToolResult(content=f"Could not write {args.path}: {e.strerror or e}", is_error=True)

        lines = args.content.count("\n") + 1
        n_bytes = len(args.content.encode("utf-8"))
        return ToolResult(content=f"Wrote {n_bytes} bytes ({lines} lines) to {args.path}")

    async def close(self) -> None:
        pass


def write_tool() -> Tool[Any]:
    return WriteTool()
