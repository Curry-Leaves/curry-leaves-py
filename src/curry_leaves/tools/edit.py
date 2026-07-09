"""The `edit` tool — targeted, exact-match text replacement in an existing file."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pydantic

from curry_leaves.core.tools import Risk, ToolResult

if TYPE_CHECKING:
    from curry_leaves.providers.base import Context


class EditArgs(pydantic.BaseModel):
    path: str = pydantic.Field(
        description="File to edit, relative to the working directory or absolute."
    )
    old_string: str = pydantic.Field(
        description="Exact text to find. Must be unique unless replace_all."
    )
    new_string: str = pydantic.Field(description="Text to replace it with.")
    replace_all: bool = pydantic.Field(
        default=False,
        description="Replace every occurrence instead of requiring a unique match.",
    )


def _count_occurrences(haystack: str, needle: str) -> int:
    if needle == "":
        return 0
    count = 0
    i = haystack.find(needle)
    while i != -1:
        count += 1
        i = haystack.find(needle, i + len(needle))
    return count


class EditTool:
    name = "edit"
    description = (
        "Replace an exact text fragment in an existing file. `old_string` must match EXACTLY "
        "(including whitespace) and be UNIQUE in the file, unless `replace_all` is true. Use this "
        "for targeted edits; use `write` to create or fully rewrite."
    )
    schema = EditArgs
    risk: Risk | None = "write"
    timeout: float | None = None

    async def run(self, args: EditArgs, ctx: "Context", signal: asyncio.Event) -> ToolResult:
        try:
            with open(args.path, encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            return ToolResult(
                content=f"File not found: {args.path} (use `write` to create it)", is_error=True
            )
        except OSError as e:
            return ToolResult(content=f"Could not read {args.path}: {e}", is_error=True)

        if args.old_string == args.new_string:
            return ToolResult(
                content="old_string and new_string are identical — nothing to do.", is_error=True
            )

        count = _count_occurrences(text, args.old_string)
        if count == 0:
            return ToolResult(
                content=f"old_string not found in {args.path} (it must match exactly).",
                is_error=True,
            )
        if count > 1 and not args.replace_all:
            return ToolResult(
                content=(
                    f"old_string is not unique in {args.path} ({count} matches). Add surrounding "
                    "context to make it unique, or pass replace_all=true."
                ),
                is_error=True,
            )

        updated = text.replace(args.old_string, args.new_string)
        try:
            with open(args.path, "w", encoding="utf-8") as f:
                f.write(updated)
        except OSError as e:
            return ToolResult(content=f"Could not write {args.path}: {e}", is_error=True)

        where = f"{count} occurrences" if args.replace_all else "1 occurrence"
        return ToolResult(content=f"Replaced {where} in {args.path}.")

    async def close(self) -> None:
        pass


def edit_tool() -> EditTool:
    return EditTool()
