#!/usr/bin/env python3
"""04_structured_output.py — get a typed object back, not just prose.

Set `output_type` to a pydantic model and the Runner injects its JSON Schema into the
prompt, validates the model's final reply against it (retrying if it isn't valid JSON),
and returns the parsed value on `result.output` — already the right shape. When the agent
has NO other tools, the provider's native JSON mode is used for a clean extraction.

Run:  python3 examples/04_structured_output.py "3 bugs, 1 P0 login crash, ship Friday"
"""

from __future__ import annotations

import asyncio
import os
import sys
from enum import Enum
from typing import cast

import pydantic

from curry_leaves import Agent, Runner

MODEL = os.environ.get("CURRY_LEAVES_MODEL", "claude-sonnet-4-5")


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# The shape we want back. `description=...` text is shown to the model as field guidance.
class StatusReport(pydantic.BaseModel):
    summary: str = pydantic.Field(description="One sentence capturing the overall state.")
    severity: Severity = pydantic.Field(description="Worst outstanding issue.")
    open_issues: int = pydantic.Field(description="Count of unresolved issues mentioned.")
    action_items: list[str] = pydantic.Field(description="Concrete next steps, imperative voice.")


async def main() -> None:
    notes = " ".join(sys.argv[1:]) or (
        "Standup: login is crashing for ~5% of users (P0), two flaky tests, docs are stale. "
        "Target ship Friday."
    )

    agent = Agent(
        model=MODEL,
        instructions="Extract a structured status report from the freeform notes.",
        output_type=StatusReport,  # no other tools -> native JSON mode + validation
    )

    result = await Runner(agent).run(notes)

    # `result.output` is already validated and typed. (`result.output_text` still has the raw JSON.)
    report = cast(StatusReport, result.output)
    print("severity :", report.severity.value)
    print("open     :", report.open_issues)
    print("summary  :", report.summary)
    print("actions  :")
    for a in report.action_items:
        print("  -", a)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:  # noqa: BLE001
        print(e, file=sys.stderr)
        sys.exit(1)
