"""Single pathogen-antibiotic scientific-interest plotting workflow."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from amr.pipeline_config import cfg, cfg_path, load_pipeline_config
from amr.pipelines.si import build_si_analysis_paths, recompute_pair_si_from_available_sources
from amr.si.core import plot_yearly_si_pair


class PairUnavailableError(ValueError):
    """Raised when no saved or recoverable data exists for a requested pair."""


def plot_single_pair_si(
    pathogen: str | Sequence[str],
    antibiotic: str | Sequence[str],
    *,
    recompute_pair_if_missing: bool = True,
    pair_recompute_min_amr_publication_count: float = 1.0,
    pair_system1_probability_threshold: float | None = None,
    pair_llm_include_decisions: list[str] | None = None,
    pair_allow_candidate_proxy: bool = True,
    project_root: str | Path | None = None,
    show: bool = True,
    _loaded_context: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Plot and save one pair, or every existing pair in two input lists.

    When either input is a sequence, the function visits the Cartesian product
    of pathogens and antibiotics. Unavailable pairs are skipped and reported.
    """
    pathogens = [pathogen] if isinstance(pathogen, str) else list(pathogen)
    antibiotics = [antibiotic] if isinstance(antibiotic, str) else list(antibiotic)
    pathogens = list(dict.fromkeys(str(value) for value in pathogens))
    antibiotics = list(dict.fromkeys(str(value) for value in antibiotics))
    is_batch = not isinstance(pathogen, str) or not isinstance(antibiotic, str)

    if project_root is None:
        project_root = Path.cwd().resolve()
    else:
        project_root = Path(project_root).resolve()
    if not (project_root / "amr").exists():
        project_root = project_root.parent
    if not (project_root / "amr").exists():
        raise RuntimeError(f"Could not find repository root from {Path.cwd()}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from IPython.display import display
    import matplotlib.pyplot as plt

    if _loaded_context is None:
        pipeline_config = load_pipeline_config(project_root=project_root)
        analysis_level = cfg(pipeline_config, "si.antibiotic_analysis_level", "antibiotic")
        si_path = build_si_analysis_paths(project_root, analysis_level)["yearly_si"]
        pathogen_background_path = cfg_path(pipeline_config, "paths.pathogen_background", project_root)
        antibiotic_background_path = cfg_path(pipeline_config, "paths.antibiotic_background", project_root)
        output_dir = cfg_path(pipeline_config, "paths.figure_dir", project_root) / "single_pair_si_plots"

        if not si_path.exists():
            raise FileNotFoundError(
                f"Missing reference SI dataset {si_path}. Run notebook 05_model_scientific_interest.ipynb first."
            )
        si_df = pd.read_parquet(si_path)
        pathogen_background_df = pd.read_parquet(pathogen_background_path)
        antibiotic_background_df = pd.read_parquet(antibiotic_background_path)
        print(f"Loaded {len(si_df):,} reference SI rows from {si_path}")
        context = (
            pipeline_config,
            si_df,
            pathogen_background_df,
            antibiotic_background_df,
            output_dir,
        )
    else:
        context = _loaded_context
        pipeline_config, si_df, pathogen_background_df, antibiotic_background_df, output_dir = context

    if is_batch:
        results: list[dict[str, Any]] = []
        skipped_pairs: list[dict[str, str]] = []
        for pathogen_name in pathogens:
            for antibiotic_name in antibiotics:
                try:
                    result = plot_single_pair_si(
                        pathogen_name,
                        antibiotic_name,
                        recompute_pair_if_missing=recompute_pair_if_missing,
                        pair_recompute_min_amr_publication_count=pair_recompute_min_amr_publication_count,
                        pair_system1_probability_threshold=pair_system1_probability_threshold,
                        pair_llm_include_decisions=pair_llm_include_decisions,
                        pair_allow_candidate_proxy=pair_allow_candidate_proxy,
                        project_root=project_root,
                        show=show,
                        _loaded_context=context,
                    )
                    results.append(result)
                except PairUnavailableError as exc:
                    skipped_pairs.append(
                        {"pathogen": pathogen_name, "antibiotic": antibiotic_name, "reason": str(exc)}
                    )
                    print(f"Skipping unavailable pair {pathogen_name} — {antibiotic_name}: {exc}")
        print(f"Generated {len(results)} plot(s); skipped {len(skipped_pairs)} unavailable pair(s).")
        return {"results": results, "skipped_pairs": skipped_pairs}

    def _saved_label(value: str, column: str) -> str:
        matches = [
            candidate
            for candidate in si_df[column].dropna().astype(str).unique()
            if candidate.casefold() == str(value).casefold()
        ]
        return matches[0] if matches else str(value)

    pathogen = _saved_label(pathogen, "pathogen")
    antibiotic = _saved_label(antibiotic, "antibiotic")
    pair_mask = (
        si_df["pathogen"].astype(str).str.casefold().eq(pathogen.casefold())
        & si_df["antibiotic"].astype(str).str.casefold().eq(antibiotic.casefold())
    )
    pair_df = si_df.loc[pair_mask].sort_values("year").copy()
    configured_min_count = float(cfg(pipeline_config, "si.min_amr_pair_publication_count"))
    configured_min_years = int(cfg(pipeline_config, "si.min_num_years_with_valid_counts"))
    fit_start_year = int(cfg(pipeline_config, "si.fit_start_year"))
    fit_end_year = int(cfg(pipeline_config, "si.fit_end_year"))
    valid_years = (
        pair_df.loc[
            pair_df["year"].between(fit_start_year, fit_end_year)
            & pair_df["count_for_model"].notna(),
            "year",
        ].nunique()
        if not pair_df.empty
        else 0
    )

    recovery = None
    recomputed_pair_si = False
    if recompute_pair_if_missing and (pair_df.empty or valid_years < configured_min_years):
        recovery = recompute_pair_si_from_available_sources(
            si_reference_df=si_df,
            pathogen=pathogen,
            antibiotic=antibiotic,
            pipeline_config=pipeline_config,
            project_root=project_root,
            pathogen_background_df=pathogen_background_df,
            antibiotic_background_df=antibiotic_background_df,
            min_amr_pair_publication_count=pair_recompute_min_amr_publication_count,
            system1_probability_threshold=pair_system1_probability_threshold,
            llm_include_decisions=pair_llm_include_decisions,
            allow_candidate_proxy=pair_allow_candidate_proxy,
        )
        display(recovery["availability_by_year"])
        if recovery["pair_si_df"].empty:
            raise PairUnavailableError(recovery["message"])
        recomputed_pair_df = recovery["pair_si_df"]
        pathogen = str(recomputed_pair_df.iloc[0]["pathogen"])
        antibiotic = str(recomputed_pair_df.iloc[0]["antibiotic"])
        pair_mask = si_df["pathogen"].eq(pathogen) & si_df["antibiotic"].eq(antibiotic)
        si_plot_df = pd.concat([si_df.loc[~pair_mask], recomputed_pair_df], ignore_index=True)
        pair_df = recomputed_pair_df.sort_values("year").copy()
        recomputed_pair_si = True
        plot_min_count = pair_recompute_min_amr_publication_count
        pair_data_label = recovery["data_label"]
        print(f"Recovered from: {recovery['source']} ({recovery['candidate_source']})")
        print(f"Interpretation: {recovery['data_interpretation']}")
    else:
        si_plot_df = si_df
        plot_min_count = configured_min_count
        pair_data_label = "AMR pair"
        print(f"Using saved SI for {pathogen} — {antibiotic}; yearly rows: {len(pair_df)}")

    if pair_df.empty:
        raise PairUnavailableError(f"No saved SI rows found for {pathogen} — {antibiotic}.")

    display(pair_df[["year", "count", "count_for_model", "SI"]].reset_index(drop=True))

    fit_start_year = int(cfg(pipeline_config, "si.fit_start_year"))
    fit_end_year = int(cfg(pipeline_config, "si.curve_fit_end_year"))
    num_future_years = int(cfg(pipeline_config, "si.num_future_years"))
    min_num_years_with_valid_counts = int(cfg(pipeline_config, "si.min_num_years_with_valid_counts"))
    if recomputed_pair_si:
        available_years = pair_df.loc[
            pair_df["year"].between(fit_start_year, fit_end_year) & pair_df["count_for_model"].notna(),
            "year",
        ].nunique()
        if available_years == 0:
            fit_start_year = int(pair_df["year"].min())
            fit_end_year = int(pair_df["year"].max())
            available_years = pair_df["year"].nunique()
        min_num_years_with_valid_counts = max(
            1,
            min(min_num_years_with_valid_counts, available_years),
        )
    loess_frac = float(cfg(pipeline_config, "si.loess_frac"))
    loess_degree = int(cfg(pipeline_config, "si.loess_degree"))
    tail_damping_years = cfg(pipeline_config, "si.tail_damping_years")
    fill_missing_amr_counts_with_zero = bool(cfg(pipeline_config, "si.fill_missing_amr_counts_with_zero"))

    plot_result = plot_yearly_si_pair(
        si_df=si_plot_df,
        pathogen=pathogen,
        antibiotic=antibiotic,
        fit_start_year=fit_start_year,
        fit_end_year=fit_end_year,
        num_additional_years=num_future_years,
        loess_frac=loess_frac,
        loess_degree=loess_degree,
        tail_damping_years=tail_damping_years,
        min_amr_pair_publication_count=plot_min_count,
        min_num_years_with_valid_counts=min_num_years_with_valid_counts,
        fill_missing_amr_counts_with_zero=fill_missing_amr_counts_with_zero,
        pathogen_background_df=pathogen_background_df,
        antibiotic_background_df=antibiotic_background_df,
        pair_count_label=pair_data_label,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    def _plot_slug(value: str) -> str:
        return str(value).strip().replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")

    output_path = output_dir / f"{_plot_slug(pathogen)}__{_plot_slug(antibiotic)}.pdf"
    if recomputed_pair_si and pair_data_label == "candidate pair":
        plot_result["ax_si"].set_ylabel("Candidate-proxy residual (not validated SI)")
        plot_result["ax_si"].set_title(
            "UNSCREENED CANDIDATE PROXY — NOT AMR-VALIDATED\n" + plot_result["ax_si"].get_title()
        )
    plot_result["figure"].savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    print(
        f"Fit: {fit_start_year}–{fit_end_year}; forecast extension: {num_future_years} years; "
        f"valid training years: {plot_result['train_df']['year'].nunique()}; "
        f"SI mode: {recovery['data_interpretation'] if recomputed_pair_si else 'saved canonical data'}"
    )
    if show:
        display(plot_result["figure"])
    plt.close(plot_result["figure"])

    return {
        "pathogen": pathogen,
        "antibiotic": antibiotic,
        "pair_df": pair_df,
        "recovery": recovery,
        "plot_result": plot_result,
        "output_path": output_path,
    }


__all__ = ["plot_single_pair_si"]
