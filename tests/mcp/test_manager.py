from __future__ import annotations

import asyncio

import pytest

from curry_leaves.mcp.manager import McpServerManager


class _StubServer:
    def __init__(self, name: str, *, fail: bool = False, delay: float = 0.0) -> None:
        self.name = name
        self._fail = fail
        self._delay = delay
        self.connect_calls = 0
        self.close_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise ConnectionError(f"{self.name} refused connection")

    async def __aenter__(self) -> "_StubServer":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def list_tools(self) -> list[object]:
        return []

    async def close(self) -> None:
        self.close_calls += 1


async def test_all_succeed() -> None:
    s1, s2 = _StubServer("a"), _StubServer("b")
    async with McpServerManager([s1, s2]) as manager:
        assert {s.name for s in manager.active_servers} == {"a", "b"}
        assert manager.failed_servers == []
        assert manager.errors == {}
        assert manager.get("a") is s1
        assert manager.get("b") is s2
    assert s1.close_calls == 1
    assert s2.close_calls == 1


async def test_drops_failed_server_by_default() -> None:
    good, bad = _StubServer("good"), _StubServer("bad", fail=True)
    async with McpServerManager([good, bad]) as manager:
        assert [s.name for s in manager.active_servers] == ["good"]
        assert [s.name for s in manager.failed_servers] == ["bad"]
        assert "bad" in manager.errors
        assert isinstance(manager.errors["bad"], ConnectionError)


async def test_get_unknown_server_raises_keyerror() -> None:
    async with McpServerManager([_StubServer("a")]) as manager:
        with pytest.raises(KeyError):
            manager.get("nonexistent")


async def test_get_failed_server_raises_keyerror_with_reason() -> None:
    async with McpServerManager([_StubServer("bad", fail=True)]) as manager:
        with pytest.raises(KeyError, match="failed to connect"):
            manager.get("bad")


async def test_drop_failed_servers_false_raises() -> None:
    with pytest.raises(ConnectionError):
        async with McpServerManager([_StubServer("bad", fail=True)], drop_failed_servers=False):
            pass


async def test_close_all_closes_every_server_including_failed() -> None:
    good, bad = _StubServer("good"), _StubServer("bad", fail=True)
    async with McpServerManager([good, bad]):
        pass
    assert good.close_calls == 1
    # A server that never successfully connected still gets close() called on exit
    # (idempotent servers tolerate this; it's a best-effort cleanup pass).
    assert bad.close_calls == 1


async def test_connect_in_parallel() -> None:
    s1, s2 = _StubServer("a", delay=0.05), _StubServer("b", delay=0.05)
    loop = asyncio.get_event_loop()
    start = loop.time()
    async with McpServerManager([s1, s2], connect_in_parallel=True):
        pass
    elapsed = loop.time() - start
    assert elapsed < 0.09  # both connects overlap; well under 2x delay


async def test_reconnect_failed_only() -> None:
    bad = _StubServer("bad", fail=True)
    async with McpServerManager([bad]) as manager:
        assert manager.failed_servers == [bad]
        bad._fail = False
        await manager.reconnect(failed_only=True)
        assert manager.active_servers == [bad]
        assert manager.failed_servers == []
