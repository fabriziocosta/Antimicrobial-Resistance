"""Helpers for converting interval-based PMID results into yearly pair counts."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .search import fetch_pub_metadata


def _coerce_pmid_list(value: object) -> list[str]:
    """Normalize a stored PMID collection into a list of string PMIDs."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = [text]
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [text]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return []
    return [str(value).strip()]


def explode_positive_pair_pmids(positive_df: pd.DataFrame) -> pd.DataFrame:
    """Explode interval-level positive results to one row per pair-PMID association."""
    required_cols = {"pathogen", "antibiotic", "start_date", "end_date", "pmids"}
    missing = required_cols - set(positive_df.columns)
    if missing:
        raise KeyError(f"positive_df is missing required columns: {sorted(missing)}")

    work = positive_df.loc[:, ["pathogen", "antibiotic", "start_date", "end_date", "pmids"]].copy()
    work["pmids"] = work["pmids"].apply(_coerce_pmid_list)
    pair_pmid_df = work.explode("pmids", ignore_index=True).rename(columns={"pmids": "pmid"})
    pair_pmid_df["pmid"] = pair_pmid_df["pmid"].astype(str).str.strip()
    pair_pmid_df = pair_pmid_df.loc[pair_pmid_df["pmid"] != ""].copy()
    pair_pmid_df = pair_pmid_df.drop_duplicates(
        subset=["pathogen", "antibiotic", "pmid"],
        keep="first",
    ).reset_index(drop=True)
    return pair_pmid_df


def load_pubmed_metadata_cache(cache_path: str | Path) -> pd.DataFrame:
    """Load a PMID metadata cache from parquet or return an empty frame."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return pd.DataFrame(columns=["pmid", "pub_date", "pub_year", "venue"])

    cache_df = pd.read_parquet(cache_path)
    expected_cols = ["pmid", "pub_date", "pub_year", "venue"]
    for col in expected_cols:
        if col not in cache_df.columns:
            cache_df[col] = pd.NA
    cache_df = cache_df.loc[:, expected_cols].copy()
    cache_df["pmid"] = cache_df["pmid"].astype(str)
    cache_df["pub_date"] = pd.to_datetime(cache_df["pub_date"], errors="coerce")
    cache_df["pub_year"] = pd.to_numeric(cache_df["pub_year"], errors="coerce").astype("Int64")
    cache_df["venue"] = cache_df["venue"].fillna("Unknown").astype(str)
    cache_df = cache_df.drop_duplicates(subset=["pmid"], keep="last").reset_index(drop=True)
    return cache_df


def save_pubmed_metadata_cache(cache_df: pd.DataFrame, cache_path: str | Path) -> None:
    """Persist PMID metadata cache to parquet."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    work = cache_df.copy()
    work["pmid"] = work["pmid"].astype(str)
    work["pub_date"] = pd.to_datetime(work["pub_date"], errors="coerce")
    work["pub_year"] = pd.to_numeric(work["pub_year"], errors="coerce").astype("Int64")
    work["venue"] = work["venue"].fillna("Unknown").astype(str)
    work = work.drop_duplicates(subset=["pmid"], keep="last").sort_values("pmid").reset_index(drop=True)
    work.to_parquet(cache_path, index=False)


def refresh_pubmed_metadata_cache(
    pmids: Sequence[str],
    *,
    cache_path: str | Path,
    email: str,
    batch_size: int = 200,
    checkpoint_every_batches: int = 10,
) -> pd.DataFrame:
    """Fetch missing PubMed metadata for PMIDs and checkpoint to parquet."""
    if not email or email == "your_email@example.com":
        raise ValueError("A real PubMed email is required to fetch missing metadata.")

    cache_df = load_pubmed_metadata_cache(cache_path)
    cached_pmids = set(cache_df["pmid"].astype(str))
    unique_pmids = [str(pmid) for pmid in pd.Index(pmids).dropna().astype(str).unique()]
    missing_pmids = [pmid for pmid in unique_pmids if pmid not in cached_pmids]

    if not missing_pmids:
        print(f"PubMed metadata cache already covers all {len(unique_pmids)} requested PMIDs.")
        return cache_df

    print(
        f"Fetching PubMed metadata for {len(missing_pmids)} missing PMIDs "
        f"({len(cached_pmids)} already cached)."
    )

    new_rows: list[dict[str, object]] = []
    total_batches = (len(missing_pmids) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(missing_pmids), batch_size), start=1):
        batch = missing_pmids[start : start + batch_size]
        date_map, venue_map = fetch_pub_metadata(batch, email=email, batch_size=len(batch))
        for pmid in batch:
            pub_date = pd.to_datetime(date_map.get(pmid), errors="coerce")
            pub_year = pub_date.year if pd.notna(pub_date) else pd.NA
            new_rows.append(
                {
                    "pmid": pmid,
                    "pub_date": pub_date,
                    "pub_year": pub_year,
                    "venue": venue_map.get(pmid, "Unknown"),
                }
            )
        print(f"Fetched batch {batch_idx}/{total_batches} ({len(batch)} PMIDs).")

        if batch_idx % checkpoint_every_batches == 0 or batch_idx == total_batches:
            checkpoint_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
            save_pubmed_metadata_cache(checkpoint_df, cache_path)
            cache_df = load_pubmed_metadata_cache(cache_path)
            new_rows = []
            print(f"Checkpointed PubMed metadata cache to {cache_path}.")

    return cache_df


def attach_pubmed_metadata(
    pair_pmid_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach publication year/date metadata to pair-PMID rows."""
    required_pair_cols = {"pathogen", "antibiotic", "start_date", "end_date", "pmid"}
    missing_pair_cols = required_pair_cols - set(pair_pmid_df.columns)
    if missing_pair_cols:
        raise KeyError(f"pair_pmid_df is missing required columns: {sorted(missing_pair_cols)}")

    required_meta_cols = {"pmid", "pub_date", "pub_year", "venue"}
    missing_meta_cols = required_meta_cols - set(metadata_df.columns)
    if missing_meta_cols:
        raise KeyError(f"metadata_df is missing required columns: {sorted(missing_meta_cols)}")

    work = pair_pmid_df.copy()
    work["pmid"] = work["pmid"].astype(str)
    meta = metadata_df.loc[:, ["pmid", "pub_date", "pub_year", "venue"]].copy()
    meta["pmid"] = meta["pmid"].astype(str)
    meta["pub_date"] = pd.to_datetime(meta["pub_date"], errors="coerce")
    meta["pub_year"] = pd.to_numeric(meta["pub_year"], errors="coerce").astype("Int64")
    merged = work.merge(meta, on="pmid", how="left")
    merged["interval_start_year"] = pd.to_datetime(merged["start_date"], errors="coerce").dt.year.astype("Int64")
    merged["interval_end_year"] = pd.to_datetime(merged["end_date"], errors="coerce").dt.year.astype("Int64")
    merged["pub_year_matches_interval"] = (
        merged["pub_year"].ge(merged["interval_start_year"])
        & merged["pub_year"].le(merged["interval_end_year"])
    )
    merged.loc[merged["pub_year"].isna(), "pub_year_matches_interval"] = pd.NA
    return merged


def aggregate_pair_counts_by_year(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pair-PMID associations into yearly AMR counts."""
    required_cols = {"pathogen", "antibiotic", "pmid", "pub_year"}
    missing = required_cols - set(enriched_df.columns)
    if missing:
        raise KeyError(f"enriched_df is missing required columns: {sorted(missing)}")

    work = enriched_df.copy()
    work["pmid"] = work["pmid"].astype(str)
    work["pub_year"] = pd.to_numeric(work["pub_year"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["pub_year"]).copy()
    work["year"] = work["pub_year"].astype(int)

    yearly_df = (
        work.groupby(["pathogen", "antibiotic", "year"], as_index=False)
        .agg(
            count=("pmid", "nunique"),
            pmids=("pmid", lambda values: sorted(pd.unique(values))),
        )
        .sort_values(["pathogen", "antibiotic", "year"])
        .reset_index(drop=True)
    )
    return yearly_df


def summarize_metadata_coverage(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Return compact coverage diagnostics for fetched PubMed metadata."""
    work = enriched_df.copy()
    work["has_pub_year"] = work["pub_year"].notna()
    rows = [
        {"metric": "pair_pmid_rows", "value": int(len(work))},
        {"metric": "unique_pmids", "value": int(work["pmid"].nunique())},
        {"metric": "rows_with_pub_year", "value": int(work["has_pub_year"].sum())},
        {"metric": "rows_missing_pub_year", "value": int((~work["has_pub_year"]).sum())},
    ]
    if "pub_year_matches_interval" in work.columns:
        matched = work["pub_year_matches_interval"] == True
        mismatched = work["pub_year_matches_interval"] == False
        rows.extend(
            [
                {"metric": "rows_pub_year_matches_interval", "value": int(matched.sum())},
                {"metric": "rows_pub_year_outside_interval", "value": int(mismatched.sum())},
            ]
        )
    return pd.DataFrame(rows)


__all__ = [
    "explode_positive_pair_pmids",
    "load_pubmed_metadata_cache",
    "save_pubmed_metadata_cache",
    "refresh_pubmed_metadata_cache",
    "attach_pubmed_metadata",
    "aggregate_pair_counts_by_year",
    "summarize_metadata_coverage",
]
