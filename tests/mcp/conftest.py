from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

HTTP_FIXTURE = str(Path(__file__).parent / "fixtures" / "http_echo_server.py")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until_listening(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.1)
    raise RuntimeError(f"HTTP fixture server never started listening on port {port}")


@pytest.fixture
def http_fixture_url() -> Iterator[str]:
    """Spins up the real streamable-http fixture MCP server as a subprocess for the
    duration of one test, yields its /mcp endpoint URL, and tears it down after.
    """
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, HTTP_FIXTURE, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_until_listening(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
