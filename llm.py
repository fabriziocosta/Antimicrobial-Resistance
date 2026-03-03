"""Shim for backwards compatibility; the LLM code now lives in amr.llm."""

from __future__ import annotations

import warnings

from amr.llm import *  # noqa: F401,F403


warnings.warn(
    "llm.py has moved to amr.llm; import from amr.llm in new code.",
    category=DeprecationWarning,
    stacklevel=2,
)
