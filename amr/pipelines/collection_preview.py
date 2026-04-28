"""Notebook-facing inspection helpers for collection setup and queries."""

from __future__ import annotations

from textwrap import fill
from typing import Sequence

import pandas as pd

from amr.pipelines.collection import build_collection_preflight_summary
from amr.pubmed.queries import (
    build_antibiotic_pathogen_query,
    build_antibiotic_query,
    build_pathogen_tiab_query,
)
from amr.screening import build_user_prompt


def print_wrapped_system_prompt(system_prompt: str, *, width: int = 80) -> None:
    """Print the unified system prompt wrapped for notebook inspection."""
    print(f"PRETTY PRINT ({width} COLS) OF UNIFIED_SYSTEM_PROMPT:\n")
    print("\n".join(fill(line, width=width) if line.strip() else "" for line in system_prompt.splitlines()))


def print_pubmed_query_examples(
    *,
    pathogens: Sequence[str],
    antibiotics: Sequence[str],
    bacteria_synonym_df: pd.DataFrame,
    antibiotic_synonym_df: pd.DataFrame,
    extra_clauses: Sequence[str] | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
    max_examples: int = 3,
) -> None:
    """Print sample PubMed query clauses and final pair queries."""
    if not pathogens or not antibiotics:
        raise ValueError("At least one pathogen and one antibiotic are required for query examples.")

    sample_pathogen = pathogens[0]
    sample_antibiotic = antibiotics[0]
    sample_query = build_antibiotic_pathogen_query(
        sample_antibiotic,
        sample_pathogen,
        bacteria_synonym_df,
        antibiotic_synonym_df=antibiotic_synonym_df,
        extra_clauses=extra_clauses,
        min_date=min_date,
        max_date=max_date,
    )
    print("Sample pathogen:", sample_pathogen)
    print("Sample antibiotic:", sample_antibiotic)
    print("\nSample PubMed query:\n")
    print(sample_query)
    print("\n" + "=" * 100 + "\n")
    print("EXAMPLE PUBMED QUERIES (MULTIPLE VIEWS)\n")

    example_pairs = list(zip(pathogens[:max_examples], antibiotics[:max_examples]))
    for idx, (pathogen_name, antibiotic_name) in enumerate(example_pairs, start=1):
        pathogen_clause = build_pathogen_tiab_query(pathogen_name, bacteria_synonym_df)
        antibiotic_clause = build_antibiotic_query(
            antibiotic_name,
            antibiotic_synonym_df,
            field="Title/Abstract",
        )
        pair_query = build_antibiotic_pathogen_query(
            antibiotic_name,
            pathogen_name,
            bacteria_synonym_df,
            antibiotic_synonym_df=antibiotic_synonym_df,
            extra_clauses=extra_clauses,
            min_date=min_date,
            max_date=max_date,
        )
        print(f"Example {idx}: {pathogen_name} vs {antibiotic_name}")
        print("- Pathogen TIAB clause:")
        print(pathogen_clause)
        print("- Antibiotic TIAB clause:")
        print(antibiotic_clause)
        print("- Final pair + AMR + date query:")
        print(pair_query)
        print("\n" + "-" * 100 + "\n")


def _print_window_report(label: str, window_summary: dict) -> None:
    interval_cache = window_summary["interval_cache"]
    pathogen_bg_cache = window_summary["pathogen_bg_cache"]
    antibiotic_bg_cache = window_summary["antibiotic_bg_cache"]
    print(label)
    print(f"Date window: {window_summary['min_date']} .. {window_summary['max_date']}")
    print(f"Intervals in window: {len(window_summary['intervals'])} -> {window_summary['intervals']}")
    print(
        f"Interval cache coverage: {interval_cache['cached_rows']:,} / "
        f"{interval_cache['expected_rows']:,} rows ({interval_cache['cached_pct']:.2f}%)"
    )
    print(f"Missing interval rows still to collect: {interval_cache['missing_rows']:,}")
    print(
        f"Cached interval breadth: {interval_cache['unique_pathogens']:,} pathogens, "
        f"{interval_cache['unique_antibiotics']:,} antibiotics, "
        f"{interval_cache['unique_pairs']:,} unique pairs"
    )
    print(f"Existing yearly AMR rows in window: {window_summary['yearly_rows']:,}")
    print(f"Existing yearly pair PMID rows in window: {window_summary['yearly_pair_pmid_rows']:,}")
    print(
        f"Pathogen background cache: {pathogen_bg_cache['cached_rows']:,} / "
        f"{pathogen_bg_cache['expected_rows']:,} rows ({pathogen_bg_cache['cached_pct']:.2f}%)"
    )
    pathogen_missing_pct = (100.0 - pathogen_bg_cache["cached_pct"]) if pathogen_bg_cache["expected_rows"] else 0.0
    print(
        f"Missing pathogen background rows: {pathogen_bg_cache['missing_rows']:,} "
        f"({pathogen_missing_pct:.2f}%)"
    )
    if pathogen_bg_cache["warning"]:
        print(f"WARNING: {pathogen_bg_cache['warning']}")
    print(
        f"Antibiotic background cache: {antibiotic_bg_cache['cached_rows']:,} / "
        f"{antibiotic_bg_cache['expected_rows']:,} rows ({antibiotic_bg_cache['cached_pct']:.2f}%)"
    )
    antibiotic_missing_pct = (
        (100.0 - antibiotic_bg_cache["cached_pct"]) if antibiotic_bg_cache["expected_rows"] else 0.0
    )
    print(
        f"Missing antibiotic background rows: {antibiotic_bg_cache['missing_rows']:,} "
        f"({antibiotic_missing_pct:.2f}%)"
    )
    if antibiotic_bg_cache["warning"]:
        print(f"WARNING: {antibiotic_bg_cache['warning']}")
    print()


def print_collection_preflight(preflight: dict) -> None:
    """Print a collection preflight summary from ``build_collection_preflight_summary``."""
    print("COLLECTION PREFLIGHT\n")
    print(f"Existing yearly AMR span on disk: {preflight['yearly_year_min']} .. {preflight['yearly_year_max']}")
    print(f"Existing yearly AMR rows on disk: {preflight['yearly_rows']:,}")
    print(f"Existing yearly pair PMID rows on disk: {preflight['yearly_pair_pmid_rows']:,}\n")
    print("TOTAL WINDOW SUMMARY")
    _print_window_report("", preflight["total_window"])
    print("CURRENT SLICE SUMMARY")
    _print_window_report("", preflight["run_window"])

    interval_cache = preflight["run_window"]["interval_cache"]
    if interval_cache["missing_rows"] > 0:
        print(
            "WARNING: collection is resumable, but a non-trivial amount of interval PubMed work "
            "remains for the current slice."
        )
        if interval_cache["cached_pct"] < 50:
            print("Current-slice interval cache coverage is below 50%, so this run may still be expensive.")


def build_and_print_collection_preflight(**kwargs: object) -> dict:
    """Build and print collection preflight summary, returning the raw summary."""
    preflight = build_collection_preflight_summary(**kwargs)
    print_collection_preflight(preflight)
    return preflight


def print_unified_prompt_examples(system_prompt: str) -> None:
    """Print the system prompt and one concrete user prompt template example."""
    print("UNIFIED SYSTEM PROMPT:\n")
    print(system_prompt)
    print("\n" + "=" * 80 + "\n")
    print("USER PROMPT TEMPLATE EXAMPLE:\n")
    print(
        build_user_prompt(
            pmid="12345678",
            pathogen="Escherichia coli",
            antibiotic="Ciprofloxacin",
            title="Template title",
            abstract="Template abstract text.",
        )
    )


__all__ = [
    "build_and_print_collection_preflight",
    "print_collection_preflight",
    "print_pubmed_query_examples",
    "print_unified_prompt_examples",
    "print_wrapped_system_prompt",
]
