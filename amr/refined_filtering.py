"""Second-pass AMR relevance filtering for yearly positive PubMed records."""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd

from .filtering import _log

REFINED_AMR_SYSTEM_PROMPT = """
You are screening PubMed abstracts for relevance to antimicrobial resistance (AMR) in bacterial pathogens.

The abstract already mentions antimicrobial resistance in a bacterial pathogen species.

Decision rule:
- Answer NO if the study primarily uses the bacterial strain as a laboratory model system (for example a pharmacology assay platform, molecular biology tool, evolutionary laboratory experiment, or genetic engineering system), and the focus is not on real-world antimicrobial resistance in clinical, environmental, veterinary, food, or epidemiological contexts.
- Answer YES if the study investigates antimicrobial resistance in a real-world or applied context, including clinical infections, patient isolates, outbreaks, surveillance, epidemiology, environmental or ecological AMR studies, food microbiology, veterinary or agricultural settings, or characterisation of resistance phenotypes or genotypes in naturally occurring strains.
- If a study describes resistance in a clinical or real-world setting and also investigates the molecular basis of resistance, classify it as YES.
- Only classify as NO if the strain is primarily used as an experimental platform and AMR is incidental to the experimental system.

Return a single word only: YES or NO.
""".strip()


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _first_non_empty(series: pd.Series) -> str:
    for value in series:
        text = _normalize_text(value)
        if text:
            return text
    return ""


def build_refined_user_prompt(
    *,
    pmid: str,
    pathogen: str,
    antibiotic: str,
    year: int | str,
    title: str,
    abstract: str,
) -> str:
    """Build the user prompt for the second-pass AMR relevance screen."""
    return f"""
PMID: {pmid}
Pathogen: {pathogen}
Antibiotic: {antibiotic}
Publication year: {year}

Title:
{title or "[no title available]"}

Abstract:
{abstract or "[no abstract available]"}

Task: Decide whether this study should be included for AMR research relevance under the decision rule in the system prompt.
Respond with a single word only:
YES
or
NO
""".strip()


def prepare_yearly_positive_candidates(
    yearly_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    *,
    abstract_cache_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Explode yearly positives to one row per PMID/pathogen/antibiotic/year with title/abstract text."""
    required_yearly = {"pathogen", "antibiotic", "year", "pmids"}
    missing_yearly = required_yearly - set(yearly_df.columns)
    if missing_yearly:
        raise KeyError(f"yearly_df missing required columns: {missing_yearly}")

    work = yearly_df.copy()
    work = work.explode("pmids").dropna(subset=["pmids"]).copy()
    work["pmid"] = work["pmids"].astype(str).str.extract(r"(\d+)")[0]
    work = work.dropna(subset=["pmid"])
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["year"])
    work["year"] = work["year"].astype(int)
    work = work[["pmid", "pathogen", "antibiotic", "year"]].drop_duplicates()

    filtered_work = filtered_df.copy()
    required_filtered = {"pmid", "pathogen", "antibiotic"}
    missing_filtered = required_filtered - set(filtered_work.columns)
    if missing_filtered:
        raise KeyError(f"filtered_df missing required columns: {missing_filtered}")

    filtered_work["pmid"] = filtered_work["pmid"].astype(str)
    filtered_work["pathogen"] = filtered_work["pathogen"].astype(str)
    filtered_work["antibiotic"] = filtered_work["antibiotic"].astype(str)
    for col in ["title", "abstract", "decision", "rationale"]:
        if col not in filtered_work.columns:
            filtered_work[col] = ""

    filtered_meta = (
        filtered_work.groupby(["pmid", "pathogen", "antibiotic"], as_index=False)
        .agg(
            first_pass_decision=("decision", _first_non_empty),
            first_pass_rationale=("rationale", _first_non_empty),
            title=("title", _first_non_empty),
            abstract=("abstract", _first_non_empty),
        )
    )

    merged = work.merge(filtered_meta, on=["pmid", "pathogen", "antibiotic"], how="left")

    if abstract_cache_df is not None and not abstract_cache_df.empty:
        cache = abstract_cache_df.copy()
        required_cache = {"pmid", "title", "abstract"}
        missing_cache = required_cache - set(cache.columns)
        if missing_cache:
            raise KeyError(f"abstract_cache_df missing required columns: {missing_cache}")
        cache["pmid"] = cache["pmid"].astype(str)
        cache = cache.groupby("pmid", as_index=False).agg(
            cache_title=("title", _first_non_empty),
            cache_abstract=("abstract", _first_non_empty),
        )
        merged = merged.merge(cache, on="pmid", how="left")
        merged["title"] = merged["title"].where(merged["title"].astype(str).str.len() > 0, merged["cache_title"])
        merged["abstract"] = merged["abstract"].where(
            merged["abstract"].astype(str).str.len() > 0,
            merged["cache_abstract"],
        )
        merged = merged.drop(columns=["cache_title", "cache_abstract"])

    merged["title"] = merged["title"].map(_normalize_text)
    merged["abstract"] = merged["abstract"].map(_normalize_text)
    merged["first_pass_decision"] = merged["first_pass_decision"].map(_normalize_text).str.lower()
    merged["first_pass_rationale"] = merged["first_pass_rationale"].map(_normalize_text)
    return merged.sort_values(["year", "pathogen", "antibiotic", "pmid"]).reset_index(drop=True)


def _parse_yes_no(raw: str) -> str:
    text = (raw or "").strip().upper()
    if text.startswith("YES"):
        return "yes"
    if text.startswith("NO"):
        return "no"
    first_token = text.split()[0] if text.split() else ""
    if first_token == "YES":
        return "yes"
    if first_token == "NO":
        return "no"
    return "uncertain"


def run_refined_amr_filter(
    candidate_df: pd.DataFrame,
    llm,
    *,
    system_prompt: str = REFINED_AMR_SYSTEM_PROMPT,
    output_path: str | None = None,
    resume_from: pd.DataFrame | None = None,
    save_every: int = 50,
    progress_every: int = 25,
    max_llm_calls: int | None = None,
    dry_run: bool = False,
    verbose: int = 1,
    log_path: str | None = None,
) -> pd.DataFrame:
    """Run the refined second-pass AMR relevance filter with resumable checkpoints."""
    required_cols = {"pmid", "pathogen", "antibiotic", "year", "title", "abstract"}
    missing = required_cols - set(candidate_df.columns)
    if missing:
        raise KeyError(f"candidate_df missing required columns: {missing}")

    key_cols = ["pmid", "pathogen", "antibiotic", "year"]
    existing = pd.DataFrame()
    seen_keys: set[tuple[str, str, str, int]] = set()
    if resume_from is not None and not resume_from.empty:
        existing = resume_from.copy()
        for row in existing[key_cols].itertuples(index=False):
            seen_keys.add((str(row.pmid), str(row.pathogen), str(row.antibiotic), int(row.year)))

    total = len(candidate_df)
    results: list[dict] = []
    llm_calls_used = 0

    for idx, row in enumerate(candidate_df.itertuples(index=False), start=1):
        key = (str(row.pmid), str(row.pathogen), str(row.antibiotic), int(row.year))
        if key in seen_keys:
            continue

        title = _normalize_text(row.title)
        abstract = _normalize_text(row.abstract)
        if not abstract:
            verdict = "uncertain"
            raw_response = "NO_ABSTRACT"
            rationale = "no abstract available"
        elif dry_run:
            verdict = "uncertain"
            raw_response = "DRY_RUN"
            rationale = "dry_run"
        else:
            if max_llm_calls is not None and llm_calls_used >= max_llm_calls:
                break
            prompt = build_refined_user_prompt(
                pmid=str(row.pmid),
                pathogen=str(row.pathogen),
                antibiotic=str(row.antibiotic),
                year=int(row.year),
                title=title,
                abstract=abstract,
            )
            try:
                raw_response = llm.complete(
                    prompt,
                    system_prompt=system_prompt,
                    max_tokens=8,
                    temperature=0,
                )
                verdict = _parse_yes_no(raw_response)
                rationale = ""
            except Exception as exc:  # pragma: no cover - API/network handling
                raw_response = f"ERROR: {exc}"
                verdict = "uncertain"
                rationale = str(exc)
                _log(log_path, f"Refined filter failed for PMID {row.pmid}: {exc}")
            llm_calls_used += 1

        result = {
            "pmid": str(row.pmid),
            "pathogen": str(row.pathogen),
            "antibiotic": str(row.antibiotic),
            "year": int(row.year),
            "refined_decision": verdict,
            "refined_raw_response": raw_response,
            "refined_rationale": rationale,
            "first_pass_decision": _normalize_text(getattr(row, "first_pass_decision", "")).lower(),
            "first_pass_rationale": _normalize_text(getattr(row, "first_pass_rationale", "")),
            "title": title,
            "abstract": abstract,
        }
        results.append(result)
        seen_keys.add(key)

        if output_path and save_every and len(results) % save_every == 0:
            df_tmp = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
            df_tmp = df_tmp.drop_duplicates(subset=key_cols, keep="last")
            df_tmp.to_parquet(output_path, index=False)

        if verbose >= 1 and progress_every and len(results) % progress_every == 0:
            batch_df = pd.DataFrame(results)
            yes_n = int((batch_df["refined_decision"] == "yes").sum())
            no_n = int((batch_df["refined_decision"] == "no").sum())
            uncertain_n = int((batch_df["refined_decision"] == "uncertain").sum())
            print(
                f"Processed {len(results):>6}/{total:<6} "
                f"yes:{yes_n:>6} no:{no_n:>6} uncertain:{uncertain_n:>6}"
            )

    df_out = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    if not df_out.empty:
        df_out = df_out.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    if output_path:
        df_out.to_parquet(output_path, index=False)
    return df_out


def build_refined_yearly_positive_dataset(
    yearly_df: pd.DataFrame | str | os.PathLike[str],
    refined_df: pd.DataFrame | str | os.PathLike[str],
    *,
    output_path: str | None = None,
    pair_output_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter yearly positives by refined decisions and reaggregate counts.

    ``yearly_df`` and ``refined_df`` can be either DataFrames already loaded in
    memory or parquet paths. This allows easy restart from saved artifacts.
    """
    if isinstance(yearly_df, (str, os.PathLike)):
        yearly_df = pd.read_parquet(yearly_df)
    elif not isinstance(yearly_df, pd.DataFrame):
        raise TypeError("yearly_df must be a pandas DataFrame or a parquet file path.")

    if isinstance(refined_df, (str, os.PathLike)):
        refined_df = pd.read_parquet(refined_df)
    elif not isinstance(refined_df, pd.DataFrame):
        raise TypeError("refined_df must be a pandas DataFrame or a parquet file path.")

    required_yearly = {"pathogen", "antibiotic", "year", "pmids"}
    missing_yearly = required_yearly - set(yearly_df.columns)
    if missing_yearly:
        raise KeyError(f"yearly_df missing required columns: {missing_yearly}")

    required_refined = {"pmid", "pathogen", "antibiotic", "year", "refined_decision"}
    missing_refined = required_refined - set(refined_df.columns)
    if missing_refined:
        raise KeyError(f"refined_df missing required columns: {missing_refined}")

    work = yearly_df.copy()
    work = work.explode("pmids").dropna(subset=["pmids"]).copy()
    work["pmid"] = work["pmids"].astype(str).str.extract(r"(\d+)")[0]
    work = work.dropna(subset=["pmid"])
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["year"])
    work["year"] = work["year"].astype(int)

    # Treat "uncertain" as retained evidence alongside "yes".
    keep_keys = (
        refined_df.loc[
            refined_df["refined_decision"].astype(str).str.lower().isin({"yes", "uncertain"}),
            ["pmid", "pathogen", "antibiotic", "year"],
        ]
        .drop_duplicates()
        .astype({"pmid": str, "pathogen": str, "antibiotic": str, "year": int})
    )
    if keep_keys.empty:
        pair_df = pd.DataFrame(columns=["pathogen", "antibiotic", "year", "pmid"])
        yearly_out = pd.DataFrame(columns=["pathogen", "antibiotic", "year", "count", "pmids"])
    else:
        filtered_pairs = work.merge(keep_keys, on=["pmid", "pathogen", "antibiotic", "year"], how="inner")
        pair_df = filtered_pairs[["pathogen", "antibiotic", "year", "pmid"]].drop_duplicates().reset_index(drop=True)
        yearly_out = (
            pair_df.groupby(["pathogen", "antibiotic", "year"], as_index=False)
            .agg(pmids=("pmid", lambda s: list(dict.fromkeys(s.astype(str).tolist()))), count=("pmid", "nunique"))
            .sort_values(["pathogen", "antibiotic", "year"])
            .reset_index(drop=True)
        )

    if pair_output_path:
        pair_df.to_parquet(pair_output_path, index=False)
    if output_path:
        yearly_out.to_parquet(output_path, index=False)
    return pair_df, yearly_out


__all__ = [
    "REFINED_AMR_SYSTEM_PROMPT",
    "build_refined_user_prompt",
    "prepare_yearly_positive_candidates",
    "run_refined_amr_filter",
    "build_refined_yearly_positive_dataset",
]
