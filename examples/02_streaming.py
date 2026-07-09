#!/usr/bin/env python3
"""02_streaming.py — watch the agent think and act in real time.

`run()` waits for the final answer. `stream()` yields the loop's live AgentEvents as they
happen, so a frontend can render tokens as they arrive, show tool calls, and group turns.
Every CLI in this repo is built on exactly this stream — it is the one seam between the
engine and any UI.

The events you care about most:
  - message_update — an assistant message grew; `delta` is the incremental text chunk
  - tool_start / tool_end — a tool call began / finished
  - turn_start / turn_end — one model call + its tools (for grouping)

Run:  python3 examples/02_streaming.py "Find the largest source file."
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from curry_leaves import Agent, Runner, coding_tools

MODEL = os.environ.get("CURRY_LEAVES_MODEL", "claude-sonnet-4-5")


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or (
        "List the Python files under src/ and say what the biggest one does."
    )

    agent = Agent(
        model=MODEL,
        instructions="You are a concise coding assistant.",
        tools=coding_tools(),
    )

    runner = Runner(agent)

    async for event in runner.stream(prompt):
        if event.type == "message_update":
            # `delta` is the just-arrived slice; print it with no newline for a live typewriter.
            if event.delta is not None and event.delta.kind == "text":
                sys.stdout.write(event.delta.value)
                sys.stdout.flush()
        elif event.type == "tool_start":
            sys.stdout.write(f"\n  ▶ {event.tool_name}({json.dumps(event.args)})\n")
        elif event.type == "tool_end":
            ok = not event.result.is_error
            sys.stdout.write(f"  {'✓' if ok else '✗'} {event.tool_name}\n")
        elif event.type == "thinking":
            sys.stdout.write(f"  🧠 reasoning effort: {event.effort}\n")
    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:  # noqa: BLE001
        print(e, file=sys.stderr)
        sys.exit(1)
