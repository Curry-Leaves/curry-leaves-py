#!/usr/bin/env python3
"""03_custom_tool.py — give an agent a tool of your own.

A Tool is just DATA + a callable: a `name`, a `description`, a pydantic `schema` (which
becomes the JSON Schema the model sees — for free via `.model_json_schema()`), a `risk`,
and a `run`. There's no `defineTool` helper in this port (see core/tools.py) — a tool is a
plain class that structurally satisfies the `Tool` protocol: just give it the right
attributes and an async `run` method.

Here we build a toy "word count" tool and a "roll dice" tool, hand them to an agent
alongside the standard read-only tools, and let the model decide when to call them.

Run:  python3 examples/03_custom_tool.py "Roll 2d6, then tell me how many words are in README.md"
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
from typing import Optional

import pydantic

from curry_leaves import Agent, Runner, find_tool, read_tool
from curry_leaves.core.tools import Risk, ToolResult
from curry_leaves.providers.base import Context

MODEL = os.environ.get("CURRY_LEAVES_MODEL", "claude-sonnet-4-5")


class RollDiceArgs(pydantic.BaseModel):
    count: int = pydantic.Field(default=1, ge=1, le=20, description="How many dice to roll.")
    sides: int = pydantic.Field(default=6, ge=2, le=100, description="Faces per die.")


class RollDiceTool:
    """A pure, side-effect-free tool -> `risk = "read"` (auto-allowed by the permission gate)."""

    name = "roll_dice"
    description = (
        "Roll `count` dice each with `sides` faces and return the individual rolls and their sum."
    )
    schema: type[pydantic.BaseModel] = RollDiceArgs
    risk: Optional[Risk] = "read"
    timeout: Optional[float] = None

    async def run(self, args: pydantic.BaseModel, ctx: Context, signal: asyncio.Event) -> ToolResult:
        assert isinstance(args, RollDiceArgs)
        rolls = [random.randint(1, args.sides) for _ in range(args.count)]
        return ToolResult(content=f"rolls={rolls} sum={sum(rolls)}")


class WordCountArgs(pydantic.BaseModel):
    path: str = pydantic.Field(description="Path to the file to count.")


class WordCountTool:
    """A tool that reads the filesystem — still `read` risk, but demonstrates real work."""

    name = "word_count"
    description = "Count the words in a UTF-8 text file at `path` (relative to the working directory)."
    schema: type[pydantic.BaseModel] = WordCountArgs
    risk: Optional[Risk] = "read"
    timeout: Optional[float] = None

    async def run(self, args: pydantic.BaseModel, ctx: Context, signal: asyncio.Event) -> ToolResult:
        assert isinstance(args, WordCountArgs)
        try:
            text = await asyncio.to_thread(lambda: open(args.path, encoding="utf-8").read())
        except OSError as e:
            return ToolResult(content=f"could not read {args.path}: {e}", is_error=True)
        words = len(text.split())
        return ToolResult(content=f"{words} words in {args.path}")


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Roll 2d6, then tell me how many words README.md has."

    agent = Agent(
        model=MODEL,
        instructions="You are a helpful assistant. Use the provided tools rather than guessing.",
        tools=[RollDiceTool(), WordCountTool(), read_tool(), find_tool()],
    )

    result = await Runner(agent).run(prompt)
    print(result.output_text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:  # noqa: BLE001
        print(e, file=sys.stderr)
        sys.exit(1)
