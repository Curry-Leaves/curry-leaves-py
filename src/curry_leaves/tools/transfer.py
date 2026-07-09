"""`transfer` — hand off the WHOLE conversation to another agent; it takes over and the
current agent does NOT resume. Contrast with `task` (delegation returns a result to the
caller). Description is built dynamically from the roster.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable, Optional

import pydantic

from curry_leaves.core.tools import Risk, ToolResult
from curry_leaves.providers.base import Context

if TYPE_CHECKING:
    from curry_leaves.core.agent import Agent

# Queues a one-way handoff on the Runner; returns a confirmation/error string.
Transfer = Callable[[str], str]


class TransferArgs(pydantic.BaseModel):
    agent: str = pydantic.Field(
        description="Which agent to hand the conversation off to (one of the names listed)."
    )


class TransferTool:
    """Structurally satisfies the `Tool` protocol (see core/tools.py)."""

    name = "transfer"
    risk: Optional[Risk] = "read"
    schema: type[pydantic.BaseModel] = TransferArgs
    timeout: Optional[float] = None

    def __init__(self, agents: "dict[str, Agent]", transfer: Transfer) -> None:
        self._transfer = transfer
        listing = "\n".join(
            f"- {a.name}: {a.description or 'specialist agent'}" for a in agents.values()
        )
        self.description = (
            "Hand off the WHOLE conversation to another agent — it takes over and continues from here; "
            "you do NOT resume. Use for routing/triage when the request belongs to a different agent. "
            "(For a bounded sub-task you want a result from, call that agent's tool instead.)\nAgents:\n"
            + listing
        )

    async def run(self, args: TransferArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        return ToolResult(content=self._transfer(args.agent))

    async def close(self) -> None:
        pass
