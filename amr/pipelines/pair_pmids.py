"""Helpers for inspecting PMID evidence behind a pathogen-antibiotic pair."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PUBMED_BASE_URL = "https://pubmed.ncbi.nlm.nih.gov"


def _clean_entity(value: object) -> str:
    return str(value).strip()


def slugify_output_name(value: object) -> str:
    """Return a filesystem-friendly lowercase name fragment."""
    return (
        str(value)
        .strip()
        .lower()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _pmid_from_value(value: object) -> str | None:
    extracted = pd.Series([value]).astype(str).str.extract(r"(\d+)")[0].iloc[0]
    if pd.isna(extracted):
        return None
    return str(extracted).strip() or None


def _coerce_pmid_list(value: object) -> list[str]:
    if isinstance(value, (str, bytes)) or value is None:
        pmid = _pmid_from_value(value)
        return [] if pmid is None else [pmid]
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    elif hasattr(value, "tolist"):
        items = value.tolist()
        raw_items = items if isinstance(items, list) else []
    else:
        raw_items = []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        pmid = _pmid_from_value(item)
        if pmid and pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
    return out


def _load_table(table_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(table_or_path, pd.DataFrame):
        return table_or_path.copy()
    return pd.read_parquet(table_or_path)


def build_pair_pmid_title_table(
    yearly_screened_table: pd.DataFrame | str | Path,
    abstract_table: pd.DataFrame | str | Path,
    *,
    pathogen: str,
    antibiotic: str,
    max_pmids_per_year: int = 10,
) -> pd.DataFrame:
    """Return one row per accepted PMID for a selected pair, capped within year."""
    if max_pmids_per_year < 1:
        raise ValueError("max_pmids_per_year must be >= 1.")

    yearly_df = _load_table(yearly_screened_table)
    abstract_df = _load_table(abstract_table)
    required_yearly = {"pathogen", "antibiotic", "year", "pmids"}
    missing_yearly = required_yearly - set(yearly_df.columns)
    if missing_yearly:
        raise KeyError(f"yearly_screened_table is missing columns: {sorted(missing_yearly)}")
    if "pmid" not in abstract_df.columns:
        raise KeyError("abstract_table must contain a 'pmid' column.")
    if "title" not in abstract_df.columns:
        abstract_df = abstract_df.copy()
        abstract_df["title"] = ""

    target_pathogen = _clean_entity(pathogen).casefold()
    target_antibiotic = _clean_entity(antibiotic).casefold()
    work = yearly_df.loc[
        yearly_df["pathogen"].map(lambda value: _clean_entity(value).casefold()).eq(target_pathogen)
        & yearly_df["antibiotic"].map(lambda value: _clean_entity(value).casefold()).eq(target_antibiotic)
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["year", "PMID", "title", "pmid", "pmid_url", "pathogen", "antibiotic"])

    work["pmid"] = work["pmids"].map(_coerce_pmid_list)
    work = work.explode("pmid").dropna(subset=["pmid"]).copy()
    work["pmid"] = work["pmid"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["year"]).copy()
    work["year"] = work["year"].astype(int)
    work = work.drop_duplicates(subset=["year", "pmid", "pathogen", "antibiotic"])
    work = work.sort_values(["year", "pmid"]).groupby("year", group_keys=False).head(int(max_pmids_per_year))

    abstract_df = abstract_df.copy()
    abstract_df["pmid"] = abstract_df["pmid"].map(_pmid_from_value)
    title_df = (
        abstract_df.dropna(subset=["pmid"])
        .assign(pmid=lambda df: df["pmid"].astype(str), title=lambda df: df["title"].fillna("").astype(str))
        .drop_duplicates(subset=["pmid"])
        [["pmid", "title"]]
    )
    out = work[["year", "pmid", "pathogen", "antibiotic"]].merge(title_df, on="pmid", how="left")
    out["title"] = out["title"].fillna("")
    out["pmid_url"] = out["pmid"].map(lambda pmid: f"{PUBMED_BASE_URL}/{pmid}/")
    out["PMID"] = out.apply(lambda row: f'<a href="{row.pmid_url}" target="_blank">{row.pmid}</a>', axis=1)
    return out[["year", "PMID", "title", "pmid", "pmid_url", "pathogen", "antibiotic"]].sort_values(
        ["year", "pmid"]
    ).reset_index(drop=True)


def pair_pmid_table_html(df: pd.DataFrame) -> str:
    """Render the PMID table as HTML with clickable PMID links."""
    display_cols = ["year", "PMID", "title"]
    cols = [col for col in display_cols if col in df.columns]
    return df[cols].to_html(escape=False, index=False)


def save_pair_pmid_excel(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the PMID table with a clickable Excel PMID hyperlink column."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    excel_df = df.copy()
    required_cols = {"pmid", "pmid_url"}
    missing = required_cols - set(excel_df.columns)
    if missing:
        raise KeyError(f"df is missing columns required for Excel hyperlinks: {sorted(missing)}")
    excel_df["PMID"] = excel_df.apply(lambda row: f'=HYPERLINK("{row.pmid_url}", "{row.pmid}")', axis=1)
    output_cols = ["year", "PMID", "title", "pmid", "pmid_url", "pathogen", "antibiotic"]
    excel_df[[col for col in output_cols if col in excel_df.columns]].to_excel(output_path, index=False)
    return output_path


__all__ = ["build_pair_pmid_title_table", "pair_pmid_table_html", "save_pair_pmid_excel", "slugify_output_name"]
