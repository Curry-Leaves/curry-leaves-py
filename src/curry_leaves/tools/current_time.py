"""The `current_time` tool: get the current date/time, optionally in an IANA timezone."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pydantic

from curry_leaves.core.tools import Risk, Tool, ToolResult

if TYPE_CHECKING:
    from curry_leaves.providers.base import Context


class CurrentTimeArgs(pydantic.BaseModel):
    timezone: str | None = pydantic.Field(
        default=None, description="IANA timezone name, e.g. 'Asia/Tokyo'. Omit for UTC."
    )


def _format(now: datetime, tz: ZoneInfo) -> str:
    # "YYYY-MM-DD HH:MM:SS ZZZ (UTC±HH:MM)" — assembled from the aware datetime's parts,
    # mirroring the Intl.DateTimeFormat part-assembly on the TS side.
    localized = now.astimezone(tz)
    date = localized.strftime("%Y-%m-%d")
    time = localized.strftime("%H:%M:%S")
    zone = localized.tzname() or ""
    return f"{date} {time} {zone}"


class CurrentTimeTool:
    name = "current_time"
    description = (
        "Get the current date and time. Optionally pass an IANA timezone name (e.g. 'Asia/Kolkata', "
        "'America/New_York'); defaults to UTC."
    )
    schema: type[pydantic.BaseModel] = CurrentTimeArgs
    risk: Risk | None = "read"
    timeout: float | None = None

    async def run(self, args: CurrentTimeArgs, ctx: "Context", signal: asyncio.Event) -> ToolResult:
        tz_name = args.timezone if args.timezone is not None else "UTC"
        try:
            # Validate the zone (raises ZoneInfoNotFoundError for an unknown IANA name).
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            return ToolResult(content=f"Unknown timezone: '{tz_name}'", is_error=True)
        return ToolResult(content=_format(datetime.now(timezone.utc), tz))

    async def close(self) -> None:
        pass


def current_time_tool() -> Tool[Any]:
    return CurrentTimeTool()
