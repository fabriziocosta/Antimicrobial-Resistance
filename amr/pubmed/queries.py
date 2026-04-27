"""Query construction and synonym handling for PubMed searches."""

from __future__ import annotations

import re
from typing import List

import pandas as pd


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def prepare_antibiotic_pubmed_query(name: str) -> str:
    """Build a PubMed query clause for an antibiotic name (handles hyphens/slashes)."""
    parts = re.split(r"[-/]", name)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        and_part = " AND ".join(f'"{p}"[Abstract]' for p in parts)
        query = f'("{name}"[Abstract] OR ({and_part}))'
    else:
        query = f'"{name}"[Abstract]'
    return query


def prepare_antibiotic_pubmed_query_tiab(name: str) -> str:
    """Build a PubMed query clause for an antibiotic name in Title/Abstract (handles hyphens/slashes)."""
    parts = re.split(r"[-/]", name)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        and_part = " AND ".join(f'"{p}"[Title/Abstract]' for p in parts)
        query = f'("{name}"[Title/Abstract] OR ({and_part}))'
    else:
        query = f'"{name}"[Title/Abstract]'
    return query


def expand_synonyms(pathogen: str, bacteria_synonym_df: pd.DataFrame) -> List[str]:
    """Return a list of synonyms for `pathogen` from the lookup table."""
    mask = bacteria_synonym_df["bacteria name"].str.lower() == pathogen.lower()
    vals = (
        bacteria_synonym_df.loc[mask, "bacteria synonym"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return sorted({v for v in vals if v})


def expand_antibiotic_synonyms(antibiotic: str, antibiotic_synonym_df: pd.DataFrame | None = None) -> List[str]:
    """Return canonical antibiotic plus spelling/synonym variants."""
    if antibiotic_synonym_df is None or antibiotic_synonym_df.empty:
        return [antibiotic]
    required = {"antibiotic synonym", "antibiotic name"}
    missing = required - set(antibiotic_synonym_df.columns)
    if missing:
        raise KeyError(f"antibiotic_synonym_df missing required columns: {sorted(missing)}")

    antibiotic_norm = str(antibiotic).strip().lower()
    names = antibiotic_synonym_df["antibiotic name"].astype(str).str.strip()
    synonyms = antibiotic_synonym_df["antibiotic synonym"].astype(str).str.strip()
    mask = names.str.lower().eq(antibiotic_norm)
    variants = [antibiotic]
    variants.extend(synonyms.loc[mask].dropna().tolist())
    return _dedupe_preserve_order(variants)


def build_antibiotic_query(
    antibiotic: str,
    antibiotic_synonym_df: pd.DataFrame | None = None,
    *,
    field: str = "Abstract",
) -> str:
    """Construct an antibiotic clause using canonical name, synonyms, and split variants."""
    variants = expand_antibiotic_synonyms(antibiotic, antibiotic_synonym_df)
    variant_clauses: list[str] = []
    for variant in variants:
        parts = re.split(r"[-/]", variant)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            and_part = " AND ".join(f'"{p}"[{field}]' for p in parts)
            variant_clauses.append(f'("{variant}"[{field}] OR ({and_part}))')
        else:
            variant_clauses.append(f'"{variant}"[{field}]')
    return "(" + " OR ".join(_dedupe_preserve_order(variant_clauses)) + ")"


def build_pathogen_tiab_query(pathogen: str, bacteria_synonym_df: pd.DataFrame) -> str:
    """Construct a Title/Abstract-only clause OR-ing pathogen name with its synonyms."""
    variants = expand_synonyms(pathogen, bacteria_synonym_df)
    if not variants:
        variants = [pathogen]
    elif pathogen not in variants:
        variants = [pathogen] + variants
    clause = "(" + " OR ".join(f'"{v}"[Title/Abstract]' for v in variants) + ")"
    return clause


def build_antibiotic_pathogen_query(
    antibiotic: str,
    pathogen: str,
    bacteria_synonym_df: pd.DataFrame,
    antibiotic_synonym_df: pd.DataFrame | None = None,
    extra_clauses: list[str] | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
) -> str:
    """Construct a PubMed query tying together antibiotic, pathogen, and optional filters."""
    pathogen_variants = expand_synonyms(pathogen, bacteria_synonym_df)
    if not pathogen_variants:
        pathogen_variants = [pathogen]
    elif pathogen not in pathogen_variants:
        pathogen_variants = [pathogen] + pathogen_variants

    pathogen_clause = "(" + " OR ".join(f'"{p}"[Abstract]' for p in pathogen_variants) + ")"
    antibiotic_clause = build_antibiotic_query(antibiotic, antibiotic_synonym_df, field="Abstract")

    term_parts = [f"({antibiotic_clause} AND {pathogen_clause})"]
    if extra_clauses:
        term_parts.extend(extra_clauses)
    if min_date and max_date:
        term_parts.append(f'("{min_date}"[Date - Publication] : "{max_date}"[Date - Publication])')
    return " AND ".join(term_parts)


__all__ = [
    "prepare_antibiotic_pubmed_query",
    "prepare_antibiotic_pubmed_query_tiab",
    "expand_synonyms",
    "expand_antibiotic_synonyms",
    "build_antibiotic_query",
    "build_pathogen_tiab_query",
    "build_antibiotic_pathogen_query",
]
