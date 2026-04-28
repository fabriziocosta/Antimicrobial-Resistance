"""Compatibility shim for LLM screening utilities.

New code should import from ``amr.screening``.
"""

from amr.screening.core import *  # noqa: F403
from amr.screening.core import __all__ as _core_all

__all__ = list(_core_all)
