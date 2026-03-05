"""Venue-based feature helpers."""

from __future__ import annotations

import json
from typing import Dict

import pandas as pd


def add_venue_vectors(df: pd.DataFrame, histogram_col: str = "venue_histogram"):
    """Return a copy of df with a `venue_vector` column and the venue index map."""
    if histogram_col not in df.columns:
        raise KeyError(f"Column '{histogram_col}' not present in DataFrame")

    hist_series = df[histogram_col].fillna("{}")

    def _coerce_hist(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return {}
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return {}

    parsed = hist_series.apply(_coerce_hist)
    all_venues = sorted({venue for hist in parsed for venue in hist.keys()})
    venue_index = {venue: idx for idx, venue in enumerate(all_venues)}

    def _vectorize(hist):
        vector = [0] * len(all_venues)
        for venue, count in hist.items():
            idx = venue_index.get(venue)
            if idx is not None:
                vector[idx] = count
        return vector

    venue_vectors = parsed.apply(_vectorize)

    df_copy = df.copy()
    df_copy["venue_vector"] = venue_vectors
    return df_copy, venue_index


__all__ = [
    "add_venue_vectors",
]

