"""Builtin specialist subagents — reusable Agent DEFINITIONS a parent delegates to via
the `task` tool. They run in their own isolated context. The `plan` agent returns a
STRUCTURED Plan via output_type.

    main = Agent(model=model, subagents=[explore_agent(model), plan_agent(model)])
"""

from __future__ import annotations

import pydantic

from curry_leaves.core.agent import Agent
from curry_leaves.providers.base import Model
from curry_leaves.tools.bash import bash_tool
from curry_leaves.tools.find import find_tool
from curry_leaves.tools.read import read_tool

EXPLORE_INSTRUCTIONS = """You are a fast, read-only code explorer.

Investigate the request using read-only tools (`read`, `find`). Hunt down the specific
files, functions, and patterns that matter — do not modify anything.

Return a concise, specific summary grounded in what you actually read: name the files and
symbols, and explain how the relevant pieces connect. Lead with the answer."""

PLAN_INSTRUCTIONS = """You are a software architect producing an implementation plan. Operate READ-ONLY — you
NEVER write, edit, or run state-changing commands. Investigate, then return a structured plan.

## The bar

The plan is an EXECUTION SPEC, not a design doc: a competent implementer who never saw this
conversation executes it top to bottom and makes ZERO design decisions. Detail exists to remove
the implementer's choices, not to look thorough. Do not implement — produce the plan only.

## Ground every claim

Every path, symbol, signature, and behavior you state as fact MUST come from something you actually
read this session — find it with `read`/`find`, NEVER guess. Hunt for existing functions, utilities,
and conventions to reuse BEFORE proposing anything new. Mark anything unconfirmed as `unverified`.

## Workflow

1. Understand the literal ask and the intended end state.
2. Explore the real code: read the files you'll touch, trace the data flow, find the types involved.
3. Design one approach, weigh tradeoffs briefly, commit.
4. Return the structured plan."""


class PlanStep(pydantic.BaseModel):
    description: str = pydantic.Field(
        description=(
            "The concrete edit: verb + exact target (file + symbol) + the new behavior. Name existing "
            "functions to reuse (with paths); give exact signatures/literals for new or changed symbols; "
            "state edge/failure handling. Never just 'update X'."
        )
    )
    files: list[str] = pydantic.Field(default_factory=list, description="Files this step touches.")


class Plan(pydantic.BaseModel):
    summary: str = pydantic.Field(
        description="Context: the literal ask, why it's needed, and the end state (2-4 sentences)."
    )
    steps: list[PlanStep] = pydantic.Field(
        description="Ordered steps, grouped by behavior (never one-per-file)."
    )
    critical_files: list[str] = pydantic.Field(
        default_factory=list,
        description="The <=5 files the implementer must read first, each as 'path — symbol — reason'.",
    )
    verification: str = pydantic.Field(
        description=(
            "Exact commands to prove it works end to end, including at least one check that exercises the "
            "NEW behavior (concrete input -> expected output), not only build/typecheck."
        )
    )
    assumptions: list[str] = pydantic.Field(
        default_factory=list,
        description="Only decisions you made that the user might override; pre-decide a fallback for each.",
    )


def explore_agent(model: "Model | str") -> Agent:
    """A read-only code explorer; returns a concise text summary."""
    return Agent(
        model,
        name="explore",
        description="Read-only code explorer; investigates the codebase and returns a concise summary.",
        instructions=EXPLORE_INSTRUCTIONS,
        tools=[read_tool(), find_tool(), bash_tool()],
    )


def plan_agent(model: "Model | str") -> Agent:
    """A read-only software architect; returns a STRUCTURED implementation Plan."""
    return Agent(
        model,
        name="plan",
        description="Software architect; investigates read-only and returns a structured implementation plan.",
        instructions=PLAN_INSTRUCTIONS,
        tools=[read_tool(), find_tool()],
        output_type=Plan,
    )
