"""`McpTool` — the `Tool` protocol adapter for one remote MCP tool, plus the
JSON-Schema-to-pydantic bridge its `schema` attribute needs.

`Tool.schema` (see `core/tools.py`) must be a `type[pydantic.BaseModel]` — the registry
calls `.model_json_schema()` and `.model_validate()` on it. An MCP tool's `inputSchema`
is already raw JSON Schema, so this module walks it and builds an equivalent pydantic
model dynamically via `pydantic.create_model`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

import pydantic

from curry_leaves.core.tools import Risk, ToolResult

if TYPE_CHECKING:
    from mcp import types

    from curry_leaves.mcp.client import _McpSession
    from curry_leaves.providers.base import Context

_JSON_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _field_type(prop: dict[str, Any]) -> Any:
    if "enum" in prop:
        # A Literal of the enum's values keeps validation tight without a full JSON
        # Schema $ref/oneOf compiler.
        from typing import Literal

        values = tuple(prop["enum"])
        if values:
            return Literal[values]
        return str
    json_type = prop.get("type")
    if isinstance(json_type, list):
        # e.g. ["string", "null"] -> just take the first concrete type; Optional-ness
        # is handled by the required/default logic in _pydantic_model_from_json_schema.
        json_type = next((t for t in json_type if t != "null"), None)
    if not isinstance(json_type, str):
        return Any
    return _JSON_TYPE_MAP.get(json_type, Any)


def _pydantic_model_from_json_schema(
    name: str, schema: dict[str, Any]
) -> type[pydantic.BaseModel]:
    """Build a pydantic model from a JSON Schema object (as returned by an MCP tool's
    `inputSchema`). Covers the common cases (flat properties of the basic JSON types,
    enums, required vs optional/defaulted); anything exotic ($ref, oneOf/anyOf, nested
    objects beyond a passthrough dict) falls back to a permissive `Any`-typed field
    rather than failing to build a schema at all.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = set(schema.get("required") or [])

    fields: dict[str, Any] = {}
    for prop_name, raw_prop in properties.items():
        prop = raw_prop if isinstance(raw_prop, dict) else {}
        py_type = _field_type(prop)
        description = prop.get("description")
        field_kwargs: dict[str, Any] = {}
        if description:
            field_kwargs["description"] = description

        if prop_name in required:
            fields[prop_name] = (
                py_type,
                pydantic.Field(**field_kwargs) if field_kwargs else ...,
            )
        else:
            default = prop.get("default", None)
            fields[prop_name] = (
                Optional[py_type],
                pydantic.Field(default=default, **field_kwargs),
            )

    model: type[pydantic.BaseModel] = pydantic.create_model(name, **fields)
    return model


def _flatten_content_blocks(content: list[Any]) -> str:
    """Join an MCP `CallToolResult.content` list of content blocks down to text. Text
    blocks pass through as-is; anything else (image, resource, etc.) is summarized by
    type since we can't render it as text.
    """
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(getattr(block, "text", ""))
        else:
            parts.append(f"[{block_type or 'unknown'} content omitted]")
    return "\n".join(parts) if parts else "(no output)"


class McpTool:
    """Structurally satisfies the `Tool` protocol (see core/tools.py). Wraps ONE remote
    tool on ONE connected server's session.
    """

    def __init__(
        self,
        session: "_McpSession",
        server_name: str,
        remote_tool: "types.Tool",
        *,
        risk: Risk = "exec",
        timeout: Optional[float] = None,
    ) -> None:
        self._session = session
        self._remote_name = remote_tool.name
        self.name = f"mcp__{server_name}__{remote_tool.name}"
        self.description = remote_tool.description or f"({server_name} MCP tool: {remote_tool.name})"
        self.schema: type[pydantic.BaseModel] = _pydantic_model_from_json_schema(
            f"{self.name}_Args", remote_tool.inputSchema or {}
        )
        self.risk: Optional[Risk] = risk
        self.timeout: Optional[float] = timeout

    async def run(self, args: pydantic.BaseModel, ctx: "Context", signal: asyncio.Event) -> ToolResult:
        try:
            result = await self._session.call_tool(
                self._remote_name, args.model_dump(exclude_none=True)
            )
        except Exception as e:  # noqa: BLE001 - transport/timeout/dead-session errors all map here
            return ToolResult(content=f"MCP call to '{self.name}' failed: {e}", is_error=True)
        text = _flatten_content_blocks(list(result.content))
        return ToolResult(content=text, is_error=bool(result.isError))

    async def close(self) -> None:
        await self._session.close()
