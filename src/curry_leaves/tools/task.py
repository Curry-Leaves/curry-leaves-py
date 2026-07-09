"""`task` — delegate a focused task to a specialist subagent that works in its own fresh
context and returns a result. Its description is built dynamically from the roster.
Multiple `task` calls in one turn run concurrently (parallel subagents).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

import pydantic

from curry_leaves.core.tools import Risk, ToolResult
from curry_leaves.providers.base import Context

if TYPE_CHECKING:
    from curry_leaves.core.agent import Agent

# Spawns a fresh child run for an agent definition and returns its final output text.
Spawn = Callable[["Agent", str], Awaitable[str]]


class TaskArgs(pydantic.BaseModel):
    agent: str = pydantic.Field(description="Which subagent to use (one of the names listed above).")
    prompt: str = pydantic.Field(
        description="The complete task for the subagent — it starts fresh with NO other context."
    )


class TaskTool:
    """Structurally satisfies the `Tool` protocol (see core/tools.py)."""

    name = "task"
    risk: Optional[Risk] = "read"
    schema: type[pydantic.BaseModel] = TaskArgs
    timeout: Optional[float] = None

    def __init__(self, agents: "dict[str, Agent]", spawn: Spawn) -> None:
        self._agents = agents
        self._spawn = spawn
        listing = "\n".join(
            f"- {a.name}: {a.description or 'specialist agent'}" for a in agents.values()
        )
        self.description = (
            "Delegate a focused task to a specialist subagent that works in its own fresh context and "
            "returns a result. Give it everything it needs in `prompt`.\nAvailable agents:\n" + listing
        )

    async def run(self, args: TaskArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        agent = self._agents.get(args.agent)
        if agent is None:
            names = ", ".join(self._agents.keys()) or "(none)"
            return ToolResult(content=f"Unknown agent '{args.agent}'. Available: {names}", is_error=True)
        output = await self._spawn(agent, args.prompt)
        return ToolResult(content=output or "(subagent returned no output)")

    async def close(self) -> None:
        pass
