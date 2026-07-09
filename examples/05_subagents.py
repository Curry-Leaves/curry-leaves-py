#!/usr/bin/env python3
"""05_subagents.py — one agent delegating to specialists.

List other Agents in `subagents` and the parent automatically gains two tools:
  - `task`     — delegate a sub-problem; the subagent runs in its OWN fresh context (a
                 child Runner) and returns a result the parent folds back in.
  - `transfer` — hand the conversation off entirely (a one-way switch of the active agent).

We use the two built-in specialists: `explore` (read-only investigator, returns prose) and
`plan` (architect, returns a STRUCTURED plan via its own output_type). Subagent activity
streams up to the parent's host tagged with the child's name/depth, so you can watch the
whole tree.

Run:  python3 examples/05_subagents.py "How does the permission engine decide allow vs ask?"
"""

from __future__ import annotations

import asyncio
import os
import sys

from curry_leaves import Agent, Runner, coding_tools, explore_agent, plan_agent

MODEL = os.environ.get("CURRY_LEAVES_MODEL", "claude-sonnet-4-5")


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or (
        "Investigate how tool permissions are resolved in src/, then hand me a short plan "
        "to add a `--yolo` flag."
    )

    orchestrator = Agent(
        model=MODEL,
        name="orchestrator",
        instructions=(
            "You coordinate specialists. Delegate investigation to `explore` and planning to "
            "`plan` via the `task` tool; don't do their work yourself. Synthesize their results "
            "into a final answer."
        ),
        tools=coding_tools(),
        subagents=[explore_agent(MODEL), plan_agent(MODEL)],
    )

    runner = Runner(orchestrator)

    async for event in runner.stream(prompt):
        if event.type == "message_update" and event.delta is not None and event.delta.kind == "text":
            sys.stdout.write(event.delta.value)
            sys.stdout.flush()
        elif event.type == "tool_start" and event.tool_name == "task":
            to = event.args.get("agent", "?")
            sys.stdout.write(f"\n  ⇥ delegating to '{to}'…\n")
        elif event.type == "subagent_activity" and event.event.type == "tool_start":
            # A child agent is doing work; show its tool calls indented under its name.
            sys.stdout.write(f"      [{event.name}] {event.event.tool_name}\n")
    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:  # noqa: BLE001
        print(e, file=sys.stderr)
        sys.exit(1)
