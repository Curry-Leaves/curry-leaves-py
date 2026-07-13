"""The `ask` tool: let the model put a question to the USER and await their answer."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import pydantic

from curry_leaves.core.host import AskUser
from curry_leaves.core.tools import Risk, Tool, ToolResult
from curry_leaves.providers.base import Context


class AskArgs(pydantic.BaseModel):
    question: str = pydantic.Field(description="The question, specific and self-contained.")
    options: list[str] = pydantic.Field(
        default_factory=list, description="2–4 suggested answers, if applicable."
    )


class AskTool:
    """Structurally satisfies the `Tool` protocol (see core/tools.py)."""

    name = "ask"
    risk: Optional[Risk] = "read"
    description = (
        "Ask the USER a question and wait for their answer — only for decisions you cannot settle "
        "from the code or a sensible default (a real preference or trade-off). Offer 2–4 concrete "
        "`options` when you can. Do NOT use it for things you can find yourself."
    )
    schema: type[pydantic.BaseModel] = AskArgs
    timeout: Optional[float] = None

    async def run(self, args: AskArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        if ctx.host is None:
            return ToolResult(
                content="(no interactive user available — proceed with your best judgment and note the assumption)"
            )
        req = AskUser(question=args.question, options=args.options, default="")
        answer = await ctx.host.request(req)
        return ToolResult(
            content=f"User answered: {answer}" if answer else "(user gave no answer; proceed with a default)"
        )


def ask_tool() -> Tool[Any]:
    return AskTool()
