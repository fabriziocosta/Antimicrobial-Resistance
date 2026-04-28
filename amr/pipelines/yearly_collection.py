"""Compatibility shim for yearly collection pipeline helpers.

New code should import from ``amr.pipelines.collection``.
"""

from amr.pipelines.collection import *  # noqa: F403
from amr.pipelines.collection import __all__ as _collection_all

__all__ = list(_collection_all)
