#!/usr/bin/env python3
"""06_host_and_permissions.py — the real Host + PermissionEngine, wired to your terminal.

This is the framework's frontend seam, used for real (not a toy reimplementation).

  - Host — ONE object with two halves:
      emit(event)   -> fire-and-forget progress (broadcast, no reply)
      request(req)  -> ask the user and AWAIT one typed answer
    "ask the user a question" (the `ask` tool) and "approve a risky tool call" are just two
    Request kinds on that seam. A headless run never hangs: every Request carries a `default`.

  - PermissionEngine — the gate every tool call passes through when one is supplied to the
    run. Per-call it resolves allow / ask / deny from the agent's `permissions` map + the
    tool's risk + any standing approvals. Only an `ask` verdict reaches the host. "always" /
    "session" grants are remembered so you aren't asked twice.

Point the agent at a scratch dir and ask it to write a file: `read` is auto-allowed, but the
`write` and `bash` calls are risk-gated and will prompt you at the terminal.

Run:  python3 examples/06_host_and_permissions.py "Create hello.txt with a greeting, then cat it."
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import TypeVar

from curry_leaves import (
    Agent,
    AgentEvent,
    ApproveTool,
    AskUser,
    PermissionEngine,
    Request,
    Runner,
    RunConfig,
    coding_tools,
)

MODEL = os.environ.get("CURRY_LEAVES_MODEL", "claude-sonnet-4-5")

T = TypeVar("T")


class TerminalHost:
    """A real interactive terminal Host.

    It renders progress from `emit`, and answers `request`s — both "ask the user" and
    "approve a tool" — by prompting on stdin. This is the whole contract: two methods, one
    of which returns a typed answer.
    """

    def emit(self, event: AgentEvent) -> None:
        if event.type == "message_update":
            if event.delta is not None and event.delta.kind == "text":
                sys.stdout.write(event.delta.value)
                sys.stdout.flush()
        elif event.type == "tool_start":
            sys.stdout.write(f"\n  ▶ {event.tool_name}({json.dumps(event.args)})\n")
        elif event.type == "approval":
            sys.stdout.write(f"  ⚖ {event.tool}: {'allowed' if event.granted else 'denied'} ({event.scope})\n")

    async def request(self, req: Request[T]) -> T:
        # `input()` blocks a thread, not the event loop, when run via run_in_executor —
        # mirrors the TS example's readline usage without tying up the asyncio loop.
        loop = asyncio.get_event_loop()

        if isinstance(req, AskUser):
            hint = f" [{' / '.join(req.options)}]" if req.options else ""
            answer = await loop.run_in_executor(None, input, f"\n❓ {req.question}{hint}\n> ")
            return (answer or req.default)  # type: ignore[return-value]

        if isinstance(req, ApproveTool):
            prompt = (
                f"\n⚠ allow {req.tool} [{req.risk}]? "
                "(y = once / s = session / a = always / n = deny) "
            )
            raw = (await loop.run_in_executor(None, input, prompt)).strip().lower()
            choice = "once" if raw == "y" else "session" if raw == "s" else "always" if raw == "a" else "deny"
            return choice  # type: ignore[return-value]

        return req.default


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or (
        "Create a file hello.txt containing a friendly greeting, then show me its contents."
    )

    agent = Agent(
        model=MODEL,
        instructions="You are a helpful coding assistant. Use tools to accomplish the task.",
        tools=coding_tools(),
        # Verdicts the engine honors. `read` runs freely; everything else falls back to its
        # risk (write/exec -> ask). Try {"bash": "deny"} to see a tool get blocked outright.
        permissions={"read": "allow", "find": "allow", "search": "allow"},
    )

    # Supplying a PermissionEngine is what TURNS GATING ON. Without one, every tool runs ungated.
    permission = PermissionEngine()

    runner = Runner(agent, RunConfig(host=TerminalHost(), permission=permission))

    await runner.run(prompt)
    sys.stdout.write("\n")

    grants = permission.session_grants
    if grants:
        print(f"(session grants: {', '.join(grants)})")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:  # noqa: BLE001
        print(e, file=sys.stderr)
        sys.exit(1)
