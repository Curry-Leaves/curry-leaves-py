"""The Agent — a stateless DEFINITION (config), not a stateful object.

An Agent is *what* an agent is — its model, tools, and instructions — and holds NO
conversation state. The same Agent definition can drive many conversations
concurrently, which is what makes subagents and reuse clean.

All conversation/run state (messages, steering/follow-up queues, the interrupt) lives
in the Runner, which composes an Agent and actually runs it. The loop is the pure
engine underneath.

    agent = Agent(model="claude-sonnet-4-5", instructions="…",
                   tools=[ReadTool(), WriteTool()])
    result = await Runner(agent).run("hello")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any, Literal, Optional

import pydantic

from curry_leaves.core.tools import Risk, Tool, ToolRegistry, ToolResult
from curry_leaves.providers.base import Context, Model, ModelSettings, Provider
from curry_leaves.providers.factory import infer_provider

# Per-tool permission verdicts the permission engine honors: `allow` (run), `ask`
# (prompt), `deny` (block). `"*"` is the catch-all default key.
PermissionVerdict = Literal["allow", "ask", "deny"]


@dataclass
class AgentOptions:
    """The same option set as `Agent`'s constructor, bundled as a dataclass — used
    internally by `clone()` (so `dataclasses.replace` can produce an overridden copy)
    and available to anyone who prefers to build the options separately. A dataclass
    (not pydantic) since it carries live object references (a Provider, Tool instances,
    a ToolRegistry), not pure data. The public, idiomatic way to construct an Agent is
    direct keyword arguments on `Agent(...)` itself (see below) — this mirrors the TS
    single object-literal constructor `new Agent({ model, instructions, ... })` without
    forcing Python callers through an extra options-wrapper object.
    """

    # A concrete Model, or a tier/id string the Runner resolves via preferences + catalog.
    model: "Model | str" = ""
    # The provider client. Omitted -> INFERRED from the model at construction.
    provider: Optional[Provider] = None
    name: Optional[str] = None
    # One-line description — identity for logs, and what a parent reads when delegating.
    description: Optional[str] = None
    # The agent's OWN prompt (persona/task). The Runner wraps it in the standard layers.
    instructions: Optional[str] = None
    # Always-on tools. Pass a plain list; the registry is built internally.
    tools: "list[Tool[Any]] | ToolRegistry | None" = None
    # Tools registered DEFERRED — hidden until found via `search_tools`.
    deferred_tools: Optional["list[Tool[Any]]"] = None
    model_settings: Optional[ModelSettings] = None
    max_turns: Optional[int] = None
    # Subagents this agent may delegate to (via the `task` tool), keyed by name.
    subagents: Optional[list["Agent"]] = None
    # If set, the agent must return a value matching this pydantic model. The Runner
    # injects the schema into the prompt and validates the final reply; RunResult.output
    # holds it.
    output_type: Optional[type[pydantic.BaseModel]] = None
    # Size reasoning effort per task with a tiny classifier before each run.
    auto_thinking: Optional[bool] = None
    # See PermissionVerdict. Unlisted tools fall back to their risk (read → allow, else
    # ask). Only enforced when the run is given a permission engine; otherwise ignored.
    # e.g. `{"*": "ask", "read": "allow"}`.
    permissions: Optional[dict[str, PermissionVerdict]] = None


class Agent:
    provider: Provider
    model: "Model | str"
    name: str
    description: str
    instructions: str
    tools: ToolRegistry
    model_settings: ModelSettings
    max_turns: int
    subagents: list["Agent"]
    output_type: Optional[type[pydantic.BaseModel]]
    auto_thinking: bool
    permissions: dict[str, PermissionVerdict]

    def __init__(
        self,
        model: "Model | str" = "",
        *,
        provider: Optional[Provider] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: "list[Tool[Any]] | ToolRegistry | None" = None,
        deferred_tools: "Optional[list[Tool[Any]]]" = None,
        model_settings: Optional[ModelSettings] = None,
        max_turns: Optional[int] = None,
        subagents: Optional[list["Agent"]] = None,
        output_type: Optional[type[pydantic.BaseModel]] = None,
        auto_thinking: Optional[bool] = None,
        permissions: Optional[dict[str, PermissionVerdict]] = None,
    ) -> None:
        if not model:
            raise ValueError("Agent requires a model (a Model instance or a model-id/tier string).")
        self.provider = provider if provider is not None else infer_provider(model)
        self.model = model
        self.name = name if name is not None else "agent"
        self.description = description if description is not None else ""
        self.instructions = instructions if instructions is not None else ""
        self.model_settings = model_settings if model_settings is not None else ModelSettings()
        self.max_turns = max_turns if max_turns is not None else 50
        self.subagents = list(subagents) if subagents is not None else []
        self.output_type = output_type
        self.auto_thinking = auto_thinking if auto_thinking is not None else False
        self.permissions = dict(permissions) if permissions is not None else {}

        # Normalize tools to a ToolRegistry (build one from a plain list if needed).
        if isinstance(tools, ToolRegistry):
            registry = tools
        else:
            registry = ToolRegistry()
            for t in tools or []:
                registry.register(t)
        for t in deferred_tools or []:
            registry.register(t, deferred=True)
        self.tools = registry

    def clone(self, **changes: object) -> "Agent":
        """A shallow copy with overrides — handy for spinning variants (e.g. subagents)."""
        base = AgentOptions(
            model=self.model,
            provider=self.provider,
            name=self.name,
            description=self.description,
            instructions=self.instructions,
            tools=self.tools,
            model_settings=self.model_settings,
            max_turns=self.max_turns,
            subagents=self.subagents,
            output_type=self.output_type,
            auto_thinking=self.auto_thinking,
            permissions=self.permissions,
        )
        opts = replace(base, **changes)  # type: ignore[arg-type]
        return Agent(
            opts.model,
            provider=opts.provider,
            name=opts.name,
            description=opts.description,
            instructions=opts.instructions,
            tools=opts.tools,
            deferred_tools=opts.deferred_tools,
            model_settings=opts.model_settings,
            max_turns=opts.max_turns,
            subagents=opts.subagents,
            output_type=opts.output_type,
            auto_thinking=opts.auto_thinking,
            permissions=opts.permissions,
        )

    def as_tool(self, *, name: Optional[str] = None, description: Optional[str] = None) -> "AgentTool":
        """Expose this agent as a callable Tool — the 'agent as tool' pattern (delegation).

        Drop it into another agent's `tools`; the orchestrator calls it like any function,
        the wrapped agent runs in a child Runner, and its final text comes back as the result.
        """
        return AgentTool(self, name=name, description=description)


class AgentToolArgs(pydantic.BaseModel):
    input: str = pydantic.Field(
        description="The complete task for this agent (it starts fresh, with no other context)."
    )


class AgentTool:
    """Structurally satisfies the `Tool` protocol (see core/tools.py)."""

    risk: Optional[Risk] = "read"
    schema: type[pydantic.BaseModel] = AgentToolArgs
    timeout: Optional[float] = None

    def __init__(self, agent: Agent, *, name: Optional[str] = None, description: Optional[str] = None) -> None:
        self._agent = agent
        self.name = name if name is not None else agent.name
        self.description = (
            description
            if description is not None
            else (
                f"Delegate to the '{agent.name}' agent — {agent.description or 'a specialist'}. It works "
                "in its own fresh context and returns a result; put the COMPLETE task in `input`."
            )
        )

    async def run(self, args: AgentToolArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        if ctx.spawn is None:
            return ToolResult(content=f"(cannot delegate to '{self.name}' here)", is_error=True)
        out = await ctx.spawn(self._agent, args.input)
        return ToolResult(content=out or f"({self.name} returned no output)")

    async def close(self) -> None:
        pass
