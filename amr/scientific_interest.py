"""Compatibility shim for scientific-interest utilities.

New code should import from ``amr.si``.
"""

from amr.si.core import *  # noqa: F403
from amr.si.core import __all__ as _core_all

__all__ = list(_core_all)
