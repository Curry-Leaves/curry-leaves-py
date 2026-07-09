"""ANSI escape codes for the terminal UI. Honors NO_COLOR."""

from __future__ import annotations

import os
import sys

_on = not os.environ.get("NO_COLOR") and sys.stdout.isatty()


def _code(s: str) -> str:
    return s if _on else ""


RESET = _code("\x1b[0m")
DIM = _code("\x1b[2m")
ITALIC = _code("\x1b[3m")
BOLD = _code("\x1b[1m")
CYAN = _code("\x1b[36m")
GREEN = _code("\x1b[32m")
YELLOW = _code("\x1b[33m")
RED = _code("\x1b[31m")
BLUE = _code("\x1b[34m")
MAGENTA = _code("\x1b[35m")
