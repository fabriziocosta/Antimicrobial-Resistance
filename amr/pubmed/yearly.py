"""Helpers for converting interval-based PMID results into yearly pair counts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from amr.io import atomic_write_parquet
from amr.validation import (
    clean_enriched_pair_pmid_df,
    clean_pair_pmid_df,
    clean_pubmed_metadata_df,
    clean_yearly_pair_counts,
    coerce_pmid_list,
)
from .pmids import normalize_pmids
from .search import fetch_pub_metadata


def explode_positive_pair_pmids(positive_df: pd.DataFrame) -> pd.DataFrame:
    """Explode interval-level positive results to one row per pair-PMID association."""
    required_cols = {"pathogen", "antibiotic", "start_date", "end_date", "pmids"}
    missing = required_cols - set(positive_df.columns)
    if missing:
        raise KeyError(f"positive_df is missing required columns: {sorted(missing)}")

    work = positive_df.loc[:, ["pathogen", "antibiotic", "start_date", "end_date", "pmids"]].copy()
    work["pmids"] = work["pmids"].apply(coerce_pmid_list)
    pair_pmid_df = work.explode("pmids", ignore_index=True).rename(columns={"pmids": "pmid"})
    return clean_pair_pmid_df(pair_pmid_df)


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
    return clean_pubmed_metadata_df(cache_df.loc[:, expected_cols])


def metadata_dataframe_to_dict(cache_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Convert the persisted metadata schema to the lightweight lookup schema."""
    work = clean_pubmed_metadata_df(cache_df)
    out: dict[str, dict[str, object]] = {}
    for row in work.itertuples(index=False):
        pub_date = row.pub_date
        if pd.notna(pub_date) and hasattr(pub_date, "to_pydatetime"):
            pub_date = pub_date.to_pydatetime()
        out[str(row.pmid)] = {
            "date": pub_date if pd.notna(pub_date) else None,
            "venue": str(row.venue) if pd.notna(row.venue) else "Unknown",
        }
    return out


def metadata_dict_to_dataframe(metadata_cache: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Convert the in-memory lookup schema to the persisted metadata schema."""
    rows = []
    for pmid, metadata in metadata_cache.items():
        pub_date = pd.to_datetime(metadata.get("date"), errors="coerce")
        rows.append(
            {
                "pmid": str(pmid),
                "pub_date": pub_date,
                "pub_year": int(pub_date.year) if pd.notna(pub_date) else pd.NA,
                "venue": metadata.get("venue") or "Unknown",
            }
        )
    return pd.DataFrame(rows, columns=["pmid", "pub_date", "pub_year", "venue"])


def save_pubmed_metadata_cache(cache_df: pd.DataFrame, cache_path: str | Path) -> None:
    """Persist PMID metadata cache to parquet."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    work = clean_pubmed_metadata_df(cache_df)
    atomic_write_parquet(work, cache_path, index=False)


def refresh_pubmed_metadata_cache(
    pmids: Sequence[str],
    *,
    cache_path: str | Path,
    email: str,
    batch_size: int = 1000,
    checkpoint_every_batches: int = 10,
) -> pd.DataFrame:
    """Fetch missing PubMed metadata for PMIDs and checkpoint to parquet."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if checkpoint_every_batches <= 0:
        raise ValueError("checkpoint_every_batches must be positive.")
    if not email or email == "your_email@example.com":
        raise ValueError("A real PubMed email is required to fetch missing metadata.")

    cache_df = load_pubmed_metadata_cache(cache_path)
    cached_pmids = set(cache_df["pmid"].astype(str))
    unique_pmids = normalize_pmids(pd.Index(pmids).dropna().tolist())
    missing_pmids = [pmid for pmid in unique_pmids if pmid not in cached_pmids]

    if not missing_pmids:
        print(f"PubMed metadata cache already covers all {len(unique_pmids)} requested PMIDs.")
        return cache_df

    print(
        f"Fetching PubMed metadata for {len(missing_pmids)} missing PMIDs "
        f"({len(cached_pmids)} already cached)."
    )

    new_rows: list[dict[str, object]] = []

    def _checkpoint_batch(
        batch_idx: int,
        total_batches: int,
        batch: list[str],
        date_map: dict[str, object],
        venue_map: dict[str, str],
    ) -> None:
        nonlocal cache_df, new_rows
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

        if batch_idx % checkpoint_every_batches == 0 or batch_idx == total_batches:
            new_rows_df = pd.DataFrame(new_rows)
            checkpoint_df = (
                new_rows_df
                if cache_df.empty
                else pd.concat([cache_df, new_rows_df], ignore_index=True)
            )
            save_pubmed_metadata_cache(checkpoint_df, cache_path)
            cache_df = load_pubmed_metadata_cache(cache_path)
            new_rows = []
            print(f"Checkpointed PubMed metadata cache to {cache_path}.")

    fetch_pub_metadata(
        missing_pmids,
        email=email,
        batch_size=batch_size,
        on_batch_complete=_checkpoint_batch,
    )
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

    work = clean_pair_pmid_df(pair_pmid_df)
    meta = metadata_df.loc[:, ["pmid", "pub_date", "pub_year", "venue"]].copy()
    meta = clean_pubmed_metadata_df(meta)
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

    work = clean_enriched_pair_pmid_df(enriched_df)
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
    return clean_yearly_pair_counts(yearly_df)


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
        matched = work["pub_year_matches_interval"].fillna(False).astype(bool)
        mismatched = ~matched
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
    "metadata_dataframe_to_dict",
    "metadata_dict_to_dataframe",
    "save_pubmed_metadata_cache",
    "refresh_pubmed_metadata_cache",
    "attach_pubmed_metadata",
    "aggregate_pair_counts_by_year",
    "summarize_metadata_coverage",
]
