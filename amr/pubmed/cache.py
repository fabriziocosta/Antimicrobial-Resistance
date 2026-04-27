"""Metadata cache and on-disk merge helpers for PubMed results."""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Sequence

import pandas as pd

from amr.io import atomic_write_parquet
from .search import fetch_pub_metadata


def merge_with_existing(new_df: pd.DataFrame, parquet_path: str = "pmids_results.parquet") -> pd.DataFrame:
    """Merge new results into existing parquet (if present) safely."""
    if os.path.exists(parquet_path):
        old_df = pd.read_parquet(parquet_path)
        for df in (old_df, new_df):
            if "venue_histogram" not in df.columns:
                df["venue_histogram"] = json.dumps({})
        merged_df = pd.concat([old_df, new_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(
            subset=["pathogen", "antibiotic", "start_date", "end_date"],
            keep="last",
        )
    else:
        if "venue_histogram" not in new_df.columns:
            new_df["venue_histogram"] = json.dumps({})
        merged_df = new_df

    for attempt in range(3):
        try:
            atomic_write_parquet(merged_df, parquet_path, index=False)
            break
        except OSError as e:
            delay = 2**attempt
            print(f"File write failed ({e}), retrying in {delay}s...")
            time.sleep(delay)
    return merged_df


def update_metadata_cache(pmids: Sequence[str], metadata_cache: Dict[str, dict], email: str = "your_email@example.com", batch_size: int = 200):
    """Populate metadata cache with publication date and venue details."""
    missing = [pmid for pmid in pmids if pmid not in metadata_cache]
    if not missing:
        return
    date_map, venue_map = fetch_pub_metadata(missing, email=email, batch_size=batch_size)
    for pmid in missing:
        metadata_cache[pmid] = {
            "date": date_map.get(pmid),
            "venue": venue_map.get(pmid, "Unknown"),
        }


def get_metadata(pmid: str, metadata_cache: Dict[str, dict]) -> dict:
    """Convenience accessor for cached PMID metadata."""
    return metadata_cache.get(pmid, {"date": None, "venue": "Unknown"})


__all__ = [
    "merge_with_existing",
    "update_metadata_cache",
    "get_metadata",
]
