"""Query construction and synonym handling for PubMed searches."""

from __future__ import annotations

import re
from typing import List

import pandas as pd


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
    antibiotic_clause = prepare_antibiotic_pubmed_query(antibiotic)

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
    "build_pathogen_tiab_query",
    "build_antibiotic_pathogen_query",
]

