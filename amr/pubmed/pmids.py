"""PMID normalization helpers."""

from __future__ import annotations

import math
import re

import pandas as pd

_INVALID_PMID_STRINGS = {"", "nan", "none", "null", "na", "n/a", "<na>"}


def normalize_pmid(value: object) -> str | None:
    """Return a clean PMID string, or ``None`` for missing/invalid values."""
    if value is None:
        return None
    if pd.api.types.is_scalar(value):
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if value.is_integer():
            value = int(value)
    text = str(value).strip()
    if text.lower() in _INVALID_PMID_STRINGS:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    return match.group(0)


def normalize_pmids(values: object) -> list[str]:
    """Normalize a sequence/scalar of PMID-like values and drop invalid entries."""
    if isinstance(values, (str, bytes, bytearray)) or pd.api.types.is_scalar(values):
        pmid = normalize_pmid(values)
        return [pmid] if pmid else []
    out: list[str] = []
    seen: set[str] = set()
    if values is None:
        return out
    for value in values:
        pmid = normalize_pmid(value)
        if pmid and pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
    return out


__all__ = ["normalize_pmid", "normalize_pmids"]
