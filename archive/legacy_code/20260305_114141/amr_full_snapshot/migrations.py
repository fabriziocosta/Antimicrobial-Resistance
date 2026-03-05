"""One-off migration utilities for AMR project."""

from __future__ import annotations

import pandas as pd


def backfill_filtered_to_combination_keys(
    filtered_path: str = "pmids_results_filtered.parquet",
    source_path: str = "pmids_results.parquet",
) -> tuple[int, int]:
    """
    Ensure the filtered decisions parquet keeps one row per
    (pmid, pathogen, antibiotic, start_date, end_date) combination.

    Older runs deduplicated by pmid only, which collapsed multiple
    pathogen/antibiotic pairs for the same PMID. This migration does NOT
    recreate lost per-pair decisions, but it:
    - Drops duplicates using the full combination key (keeping the latest)
    - Normalizes types (pmid/pathogen/antibiotic to str)

    Returns: (original_rows, migrated_rows)
    """
    df = pd.read_parquet(filtered_path)
    if df.empty:
        return 0, 0

    # Normalize types
    for col in ("pmid", "pathogen", "antibiotic"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    key_cols = ["pmid", "pathogen", "antibiotic", "start_date", "end_date"]
    present_keys = [c for c in key_cols if c in df.columns]
    before = len(df)

    if present_keys:
        df = df.drop_duplicates(subset=present_keys, keep="last")
    else:
        # Fallback: if combination columns are missing, we cannot safely expand
        # decisions to all pairs; just drop duplicate PMIDs to keep consistency.
        df = df.drop_duplicates(subset=["pmid"], keep="last")

    df.to_parquet(filtered_path, index=False)
    after = len(df)
    return before, after


__all__ = ["backfill_filtered_to_combination_keys"]

