from __future__ import annotations

import asyncio
from typing import Any

from mcp import types

from curry_leaves.mcp.tool import McpTool, _pydantic_model_from_json_schema


class _FakeSession:
    """A minimal stand-in for `_McpSession` — no real subprocess/network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.next_result: types.CallToolResult | None = None
        self.raise_error: Exception | None = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        self.calls.append((name, arguments))
        if self.raise_error is not None:
            raise self.raise_error
        assert self.next_result is not None
        return self.next_result

    async def close(self) -> None:
        pass


def _remote_tool(name: str, description: str = "", input_schema: dict[str, Any] | None = None) -> types.Tool:
    return types.Tool(name=name, description=description, inputSchema=input_schema or {"type": "object"})


def _text_result(text: str, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=is_error)


async def test_namespacing() -> None:
    session = _FakeSession()
    tool = McpTool(session, "github", _remote_tool("search_issues", "Search issues"))
    assert tool.name == "mcp__github__search_issues"
    assert tool.description == "Search issues"
    assert tool.risk == "exec"


async def test_run_success() -> None:
    session = _FakeSession()
    session.next_result = _text_result("found 3 issues")
    tool = McpTool(session, "github", _remote_tool("search_issues", input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }))
    args = tool.schema.model_validate({"query": "bug"})
    result = await tool.run(args, None, asyncio.Event())  # type: ignore[arg-type]
    assert result.content == "found 3 issues"
    assert result.is_error is False
    assert session.calls == [("search_issues", {"query": "bug"})]


async def test_run_maps_mcp_error_result() -> None:
    session = _FakeSession()
    session.next_result = _text_result("boom", is_error=True)
    tool = McpTool(session, "echo", _remote_tool("fail"))
    args = tool.schema.model_validate({})
    result = await tool.run(args, None, asyncio.Event())  # type: ignore[arg-type]
    assert result.is_error is True
    assert result.content == "boom"


async def test_run_maps_transport_exception() -> None:
    session = _FakeSession()
    session.raise_error = ConnectionError("pipe closed")
    tool = McpTool(session, "echo", _remote_tool("echo"))
    args = tool.schema.model_validate({})
    result = await tool.run(args, None, asyncio.Event())  # type: ignore[arg-type]
    assert result.is_error is True
    assert "pipe closed" in result.content


async def test_close_delegates_to_session() -> None:
    closed = []

    class _TrackingSession(_FakeSession):
        async def close(self) -> None:
            closed.append(True)

    session = _TrackingSession()
    tool = McpTool(session, "echo", _remote_tool("echo"))
    await tool.close()
    assert closed == [True]


def test_schema_bridge_flat_types() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "a name"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
        },
        "required": ["name", "count"],
    }
    model = _pydantic_model_from_json_schema("Args", schema)
    instance = model.model_validate({"name": "x", "count": 3})
    assert instance.name == "x"  # type: ignore[attr-defined]
    assert instance.count == 3  # type: ignore[attr-defined]
    assert instance.ratio is None  # type: ignore[attr-defined]
    assert instance.flag is None  # type: ignore[attr-defined]


def test_schema_bridge_enum() -> None:
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["a", "b", "c"]}},
        "required": ["mode"],
    }
    model = _pydantic_model_from_json_schema("Args", schema)
    instance = model.model_validate({"mode": "b"})
    assert instance.mode == "b"  # type: ignore[attr-defined]
    import pydantic

    try:
        model.model_validate({"mode": "z"})
        raised = False
    except pydantic.ValidationError:
        raised = True
    assert raised


def test_schema_bridge_array_and_object_fallback() -> None:
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array"},
            "meta": {"type": "object"},
            "anything": {},
        },
        "required": [],
    }
    model = _pydantic_model_from_json_schema("Args", schema)
    instance = model.model_validate({"tags": ["a", "b"], "meta": {"k": "v"}, "anything": 123})
    assert instance.tags == ["a", "b"]  # type: ignore[attr-defined]
    assert instance.meta == {"k": "v"}  # type: ignore[attr-defined]
    assert instance.anything == 123  # type: ignore[attr-defined]


def test_schema_bridge_no_properties() -> None:
    model = _pydantic_model_from_json_schema("Args", {"type": "object"})
    instance = model.model_validate({})
    assert instance is not None
