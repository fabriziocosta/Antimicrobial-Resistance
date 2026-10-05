"""Data-shape validation and cleanup for persisted AMR pipeline tables."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from datetime import datetime

import pandas as pd

from amr.pubmed.pmids import normalize_pmid, normalize_pmids

_BAD_TEXT_VALUES = {"", "nan", "none", "null", "na", "n/a", "<na>", "unknown"}
_INTERVAL_DATE_FORMAT = "%Y/%m/%d"


def normalize_entity_key(value: object) -> str | None:
    """Normalize canonical pathogen/antibiotic keys and reject missing-like text."""
    if value is None:
        return None
    if pd.api.types.is_scalar(value):
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
    text = " ".join(str(value).strip().split())
    if text.lower() in _BAD_TEXT_VALUES:
        return None
    return text


def coerce_pmid_list(value: object) -> list[str]:
    """Normalize a stored PMID collection into a deduplicated list of PMIDs."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                return normalize_pmids(ast.literal_eval(text))
            except (SyntaxError, ValueError):
                return normalize_pmids([text])
        return normalize_pmids([text])
    if isinstance(value, (bytes, bytearray, dict)):
        return []
    if isinstance(value, Iterable) and not pd.api.types.is_scalar(value):
        return normalize_pmids(list(value))
    return normalize_pmids([value])


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _normalize_entity_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].map(normalize_entity_key)
    return out.dropna(subset=columns).copy()


def _parse_interval_date(value: object) -> pd.Timestamp:
    parsed = pd.to_datetime(value, format=_INTERVAL_DATE_FORMAT, errors="coerce")
    return parsed


def _format_interval_date(value: object) -> object:
    parsed = _parse_interval_date(value)
    if pd.isna(parsed):
        return pd.NA
    return parsed.strftime(_INTERVAL_DATE_FORMAT)


def _current_year_upper_bound() -> int:
    return datetime.now().year + 1


def clean_interval_results(df: pd.DataFrame) -> pd.DataFrame:
    """Clean interval-level PubMed result rows before cache persistence."""
    columns = ["pathogen", "antibiotic", "start_date", "end_date", "pmids", "count", "venue_histogram"]
    work = _ensure_columns(df, columns).loc[:, columns].copy()
    work = _normalize_entity_columns(work, ["pathogen", "antibiotic"])
    work["start_date"] = work["start_date"].map(_format_interval_date)
    work["end_date"] = work["end_date"].map(_format_interval_date)
    work = work.dropna(subset=["start_date", "end_date"]).copy()
    start = pd.to_datetime(work["start_date"], format=_INTERVAL_DATE_FORMAT, errors="coerce")
    end = pd.to_datetime(work["end_date"], format=_INTERVAL_DATE_FORMAT, errors="coerce")
    work = work.loc[start.le(end)].copy()
    work["pmids"] = work["pmids"].apply(coerce_pmid_list)
    work["count"] = work["pmids"].apply(len)

    def clean_histogram(value: object) -> str:
        def normalized_items(histogram: dict) -> dict[str, int]:
            out: dict[str, int] = {}
            for key, raw_count in histogram.items():
                clean_key = normalize_entity_key(key)
                if not clean_key:
                    continue
                count = pd.to_numeric(raw_count, errors="coerce")
                if pd.isna(count):
                    continue
                out[clean_key] = max(int(count), 0)
            return out

        if isinstance(value, dict):
            return json.dumps(normalized_items(value))
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return json.dumps(normalized_items(parsed))
            except Exception:
                pass
        return json.dumps({})

    work["venue_histogram"] = work["venue_histogram"].apply(clean_histogram)
    return (
        work.drop_duplicates(subset=["pathogen", "antibiotic", "start_date", "end_date"], keep="last")
        .sort_values(["pathogen", "antibiotic", "start_date", "end_date"])
        .reset_index(drop=True)
    )


def clean_pair_pmid_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean one-row-per-pair-PMID associations."""
    columns = ["pathogen", "antibiotic", "start_date", "end_date", "pmid"]
    extra_cols = [col for col in df.columns if col not in columns]
    work = _ensure_columns(df, columns).loc[:, columns + extra_cols].copy()
    work = _normalize_entity_columns(work, ["pathogen", "antibiotic"])
    work["pmid"] = work["pmid"].map(normalize_pmid)
    work = work.dropna(subset=["pmid"]).copy()
    for col in ["start_date", "end_date"]:
        work[col] = work[col].map(_format_interval_date)
    work = work.dropna(subset=["start_date", "end_date"]).copy()
    start = pd.to_datetime(work["start_date"], format=_INTERVAL_DATE_FORMAT, errors="coerce")
    end = pd.to_datetime(work["end_date"], format=_INTERVAL_DATE_FORMAT, errors="coerce")
    work = work.loc[start.le(end)].copy()
    return (
        work.drop_duplicates(subset=["pathogen", "antibiotic", "pmid"], keep="last")
        .sort_values(["pathogen", "antibiotic", "pmid"])
        .reset_index(drop=True)
    )


def clean_pubmed_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean PMID metadata cache rows used for yearly aggregation."""
    columns = ["pmid", "pub_date", "pub_year", "venue"]
    work = _ensure_columns(df, columns).loc[:, columns].copy()
    work["pmid"] = work["pmid"].map(normalize_pmid)
    work = work.dropna(subset=["pmid"]).copy()
    work["pub_date"] = pd.to_datetime(work["pub_date"], errors="coerce")
    work["pub_year"] = pd.to_numeric(work["pub_year"], errors="coerce").astype("Int64")
    missing_year = work["pub_year"].isna() & work["pub_date"].notna()
    work.loc[missing_year, "pub_year"] = work.loc[missing_year, "pub_date"].dt.year.astype("Int64")
    work = work.dropna(subset=["pub_year"]).copy()
    work["pub_year"] = work["pub_year"].astype("Int64")
    work = work.loc[work["pub_year"].between(1900, _current_year_upper_bound())].copy()
    work["venue"] = (
        work["venue"]
        .map(lambda value: normalize_entity_key(value) if normalize_entity_key(value) else "Unknown")
        .fillna("Unknown")
    )
    return work.drop_duplicates(subset=["pmid"], keep="last").sort_values("pmid").reset_index(drop=True)


def clean_enriched_pair_pmid_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean pair-PMID rows after metadata attachment."""
    columns = [
        "pathogen",
        "antibiotic",
        "start_date",
        "end_date",
        "pmid",
        "pub_date",
        "pub_year",
        "venue",
        "interval_start_year",
        "interval_end_year",
    ]
    extra_cols = [col for col in df.columns if col not in columns]
    work = _ensure_columns(df, columns).loc[:, columns + extra_cols].copy()
    work = _normalize_entity_columns(work, ["pathogen", "antibiotic"])
    work["pmid"] = work["pmid"].map(normalize_pmid)
    work = work.dropna(subset=["pmid"]).copy()
    for col in ["start_date", "end_date"]:
        work[col] = work[col].map(_format_interval_date)
    work = work.dropna(subset=["start_date", "end_date"]).copy()
    start = pd.to_datetime(work["start_date"], format=_INTERVAL_DATE_FORMAT, errors="coerce")
    end = pd.to_datetime(work["end_date"], format=_INTERVAL_DATE_FORMAT, errors="coerce")
    work = work.loc[start.le(end)].copy()
    work["pub_date"] = pd.to_datetime(work["pub_date"], errors="coerce")
    work["pub_year"] = pd.to_numeric(work["pub_year"], errors="coerce").astype("Int64")
    missing_year = work["pub_year"].isna() & work["pub_date"].notna()
    work.loc[missing_year, "pub_year"] = work.loc[missing_year, "pub_date"].dt.year.astype("Int64")
    work = work.dropna(subset=["pub_year"]).copy()
    work["pub_year"] = work["pub_year"].astype(int)
    work = work.loc[work["pub_year"].between(1900, _current_year_upper_bound())].copy()
    work["interval_start_year"] = pd.to_numeric(work["interval_start_year"], errors="coerce").astype("Int64")
    work["interval_end_year"] = pd.to_numeric(work["interval_end_year"], errors="coerce").astype("Int64")
    has_interval_years = work["interval_start_year"].notna() & work["interval_end_year"].notna()
    work = work.loc[
        ~has_interval_years
        | (work["pub_year"].ge(work["interval_start_year"]) & work["pub_year"].le(work["interval_end_year"]))
    ].copy()
    work["venue"] = (
        work["venue"]
        .map(lambda value: normalize_entity_key(value) if normalize_entity_key(value) else "Unknown")
        .fillna("Unknown")
    )
    return (
        work.drop_duplicates(subset=["pathogen", "antibiotic", "pmid"], keep="last")
        .sort_values(["pathogen", "antibiotic", "pmid"])
        .reset_index(drop=True)
    )


def clean_yearly_pair_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Clean yearly AMR pair-count rows and enforce count/list consistency."""
    columns = ["pathogen", "antibiotic", "year", "count", "pmids"]
    extra_cols = [col for col in df.columns if col not in columns]
    work = _ensure_columns(df, columns).loc[:, columns + extra_cols].copy()
    work = _normalize_entity_columns(work, ["pathogen", "antibiotic"])
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["year"]).copy()
    work["year"] = work["year"].astype(int)
    work = work.loc[work["year"].between(1900, _current_year_upper_bound())].copy()
    work["pmids"] = work["pmids"].apply(coerce_pmid_list)
    work = work.loc[work["pmids"].apply(bool)].copy()
    work["count"] = work["pmids"].apply(len)
    return (
        work.drop_duplicates(subset=["pathogen", "antibiotic", "year"], keep="last")
        .sort_values(["pathogen", "antibiotic", "year"])
        .reset_index(drop=True)
    )


def clean_background_counts(df: pd.DataFrame, *, key_col: str) -> pd.DataFrame:
    """Clean pathogen/antibiotic yearly background-count rows."""
    columns = [key_col, "year", "count"]
    work = _ensure_columns(df, columns).loc[:, columns].copy()
    work = _normalize_entity_columns(work, [key_col])
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["count"] = pd.to_numeric(work["count"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["year", "count"]).copy()
    work["year"] = work["year"].astype(int)
    work["count"] = work["count"].astype(int).clip(lower=0)
    work = work.loc[work["year"].between(1900, _current_year_upper_bound())].copy()
    return (
        work.drop_duplicates(subset=[key_col, "year"], keep="last")
        .sort_values([key_col, "year"])
        .reset_index(drop=True)
    )


__all__ = [
    "normalize_entity_key",
    "coerce_pmid_list",
    "clean_interval_results",
    "clean_pair_pmid_df",
    "clean_pubmed_metadata_df",
    "clean_enriched_pair_pmid_df",
    "clean_yearly_pair_counts",
    "clean_background_counts",
]
