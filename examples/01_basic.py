#!/usr/bin/env python3
"""01_basic.py — the smallest real curry-leaves program: one Agent, one Runner, one prompt.

An `Agent` is a stateless DEFINITION (model + tools + instructions). A `Runner` gives it a
live conversation and drives the streaming tool-use loop to completion.

Run (after `pip install -e .`):

    export ANTHROPIC_API_KEY=sk-ant-...        # or OPENAI_API_KEY=sk-...
    python3 examples/01_basic.py "What is this project?"

Set CURRY_LEAVES_MODEL to pick a model (default: claude-sonnet-4-5).
"""

from __future__ import annotations

import asyncio
import os
import sys

from curry_leaves import Agent, Runner, coding_tools

MODEL = os.environ.get("CURRY_LEAVES_MODEL", "claude-sonnet-4-5")


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Summarize README.md in three bullets."

    agent = Agent(
        model=MODEL,
        instructions="You are a concise coding assistant.",
        tools=coding_tools(),  # read / write / edit / find / search / bash / task / ask
    )

    # `run` drives the loop until the model stops calling tools, then returns the result.
    result = await Runner(agent).run(prompt)

    print(result.output_text)
    print(f"\n— {result.usage.output} output tokens, cost ${result.usage.cost.total:.4f}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:  # noqa: BLE001 — top-level CLI error boundary, mirrors the TS catch
        print(e, file=sys.stderr)
        sys.exit(1)
