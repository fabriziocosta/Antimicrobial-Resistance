"""Helpers for the GLASS interactive scientific-interest notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from amr.glass import load_and_clean_glass_list
from amr.si import rank_yearly_pairs_by_future_si_shift

REQUIRED_INTERACTIVE_RANK_COLUMNS = {
    "latest_available_count",
    "latest_available_predicted_SI",
    "post_fit_count_abs_pct_error_mean",
    "n_count_error_years",
}

GLASS_MATCH_COLUMNS = [
    "pathogen",
    "antibiotic",
    "latest_available_year",
    "future_year",
    "latest_available_count",
    "future_predicted_count",
    "latest_smoothed_SI",
    "future_predicted_SI",
    "future_minus_latest_smoothed_SI",
    "post_fit_count_abs_pct_error_mean",
    "glass_group_column",
]

NOVEL_PREDICTION_COLUMNS = [
    "pathogen",
    "antibiotic",
    "latest_available_year",
    "future_year",
    "latest_available_count",
    "future_predicted_count",
    "latest_smoothed_SI",
    "future_predicted_SI",
    "future_minus_latest_smoothed_SI",
    "post_fit_count_abs_pct_error_mean",
]


def load_si_and_clean_glass_list(
    yearly_si_path: str | Path,
    glass_xlsx_path: str | Path,
    *,
    bacteria_synonym_path: str | Path | None = None,
    antibiotic_synonym_path: str | Path | None = None,
    antibiotic_coarsening_path: str | Path | None = None,
    glass_cleaned_path: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load yearly SI data and save a standardized GLASS pathogen-antibiotic list."""
    yearly_si_path = Path(yearly_si_path)
    si_df = pd.read_parquet(yearly_si_path)
    if verbose:
        print(f"Loaded SI rows: {len(si_df):,} from {yearly_si_path}")
        print(f"Pathogens: {si_df['pathogen'].nunique():,}; antibiotics/levels: {si_df['antibiotic'].nunique():,}")

    glass_pairs_df = load_and_clean_glass_list(
        glass_xlsx_path,
        bacteria_synonym_path=bacteria_synonym_path,
        antibiotic_synonym_path=antibiotic_synonym_path,
        antibiotic_coarsening_path=antibiotic_coarsening_path,
        output_path=glass_cleaned_path,
        pathogen_universe=si_df["pathogen"].dropna().unique(),
    )
    if verbose:
        destination = f" to {glass_cleaned_path}" if glass_cleaned_path is not None else ""
        print(f"Saved cleaned GLASS rows: {len(glass_pairs_df):,}{destination}")
    return si_df, glass_pairs_df


def load_or_build_interactive_rank(
    si_df: pd.DataFrame,
    rank_output_path: str | Path,
    *,
    source_path: str | Path | None = None,
    fit_start_year: int,
    fit_end_year: int,
    num_future_years: int,
    loess_frac: float = 0.6,
    loess_degree: int = 1,
    tail_damping_years: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load cached rank data when usable, otherwise rebuild it with interactive fields."""
    rank_output_path = Path(rank_output_path)
    if rank_output_path.exists():
        rank_df = pd.read_parquet(rank_output_path)
        missing_rank_columns = REQUIRED_INTERACTIVE_RANK_COLUMNS - set(rank_df.columns)
    else:
        rank_df = pd.DataFrame()
        missing_rank_columns = REQUIRED_INTERACTIVE_RANK_COLUMNS

    cache_is_stale = False
    if source_path is not None and rank_output_path.exists():
        source_path = Path(source_path)
        if source_path.exists():
            cache_is_stale = rank_output_path.stat().st_mtime < source_path.stat().st_mtime

    if rank_df.empty or missing_rank_columns or cache_is_stale:
        if verbose and missing_rank_columns:
            print(f"Recomputing rank data because cached file is missing: {sorted(missing_rank_columns)}")
        if verbose and cache_is_stale:
            print(f"Recomputing rank data because {rank_output_path} is older than {source_path}")
        rank_df = rank_yearly_pairs_by_future_si_shift(
            si_df,
            fit_start_year=fit_start_year,
            fit_end_year=fit_end_year,
            num_future_years=num_future_years,
            save_path=str(rank_output_path),
            loess_frac=loess_frac,
            loess_degree=loess_degree,
            tail_damping_years=tail_damping_years,
            verbose=verbose,
        )
    elif verbose:
        print(f"Loaded cached rank rows: {len(rank_df):,} from {rank_output_path}")
    return rank_df


def build_glass_match_table(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Return plotted GLASS matches sorted by latest observed AMR count."""
    return (
        plot_df.loc[plot_df["is_glass_pair"], GLASS_MATCH_COLUMNS]
        .sort_values("latest_available_count", ascending=False)
        .reset_index(drop=True)
    )


def build_novel_prediction_table(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Return plotted non-GLASS pairs sorted as candidate novel predictions."""
    return (
        plot_df.loc[~plot_df["is_glass_pair"], NOVEL_PREDICTION_COLUMNS]
        .sort_values(["future_minus_latest_smoothed_SI", "latest_available_count"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _add_plot_labels(
    plot_df: pd.DataFrame,
    *,
    display_name: bool,
    max_text_labels: int | None,
    always_label_glass: bool = True,
) -> pd.DataFrame:
    plot_df = plot_df.copy()
    plot_df["appears_in_GLASS"] = np.where(plot_df["is_glass_pair"], "yes", "no")
    plot_df["label"] = plot_df.apply(
        lambda row: f"{str(row['pathogen']).strip()[:3]}<br>{str(row['antibiotic']).strip()[:3]}",
        axis=1,
    )
    plot_df["display_label"] = ""
    if display_name:
        if max_text_labels is None:
            label_mask = pd.Series(True, index=plot_df.index)
        else:
            label_mask = pd.Series(False, index=plot_df.index)
            if always_label_glass:
                label_mask = label_mask | plot_df["is_glass_pair"].fillna(False)
            remaining = max(int(max_text_labels) - int(label_mask.sum()), 0)
            if remaining:
                extra_index = (
                    plot_df.loc[~label_mask]
                    .sort_values(["future_minus_latest_smoothed_SI", "latest_available_count"], ascending=[False, False])
                    .head(remaining)
                    .index
                )
                label_mask.loc[extra_index] = True
        plot_df.loc[label_mask, "display_label"] = plot_df.loc[label_mask, "label"]
    return plot_df


def _write_plotly_outputs(
    fig: Any,
    *,
    html_path: str | Path | None,
    pdf_path: str | Path | None,
    width: int,
    height: int,
) -> tuple[str | None, str | None, str | None]:
    saved_html_path = None
    if html_path:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        saved_html_path = str(html_path)
    saved_pdf_path = None
    pdf_error = None
    if pdf_path:
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(pdf_path), format="pdf", width=int(width), height=int(height))
            saved_pdf_path = str(pdf_path)
        except Exception as exc:  # pragma: no cover - depends on optional static export backend/browser
            if saved_html_path is None:
                fallback_html_path = pdf_path.with_suffix(".html")
                fig.write_html(str(fallback_html_path), include_plotlyjs="cdn")
                saved_html_path = str(fallback_html_path)
            pdf_error = _format_plotly_pdf_error(exc, saved_html_path)
            warnings.warn(
                pdf_error,
                RuntimeWarning,
                stacklevel=2,
            )
    return saved_pdf_path, pdf_error, saved_html_path


def _format_plotly_pdf_error(exc: Exception, html_path: str | None) -> str:
    message = str(exc)
    if "/snap/bin/chromium" in message or "browser seemed to close immediately" in message.lower():
        reason = "Kaleido could not start the Chrome/Chromium browser it found, likely the Snap Chromium install."
    elif "chrome" in message.lower() or "chromium" in message.lower():
        reason = "Kaleido could not start a working Chrome/Chromium browser."
    else:
        reason = f"Plotly PDF export failed: {message}"
    fallback = f" Saved interactive HTML fallback: {html_path}." if html_path else ""
    return f"{reason}{fallback} Install a working Chrome with `plotly_get_chrome` or `choreo_get_chrome` to enable PDF export."


def _build_glass_si_scatter(
    scatter_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    min_count_abs_pct_error_mean: float,
    max_count_abs_pct_error_mean: float,
    display_name: bool,
    max_text_labels: int | None,
    always_label_glass: bool,
    width: int,
    height: int,
    add_zero_lines: bool,
    add_identity_line: bool,
) -> tuple[Any, pd.DataFrame]:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional notebook dependency
        raise ImportError("plotly is required for interactive SI plots.") from exc

    scatter_df = _add_plot_labels(
        scatter_df.copy(),
        display_name=display_name,
        max_text_labels=max_text_labels,
        always_label_glass=always_label_glass,
    )
    show_text = display_name and scatter_df["display_label"].astype(str).str.len().gt(0).any()

    fig = go.Figure(
        data=[
            go.Scatter(
                x=scatter_df[x_col],
                y=scatter_df[y_col],
                mode="markers+text" if show_text else "markers",
                text=scatter_df["display_label"] if show_text else None,
                textposition="middle right",
                textfont=dict(size=9),
                marker=dict(
                    color=pd.to_numeric(scatter_df["post_fit_count_abs_pct_error_mean"], errors="coerce"),
                    colorscale=[[0.0, "#2166ac"], [1.0, "#ffffff"]],
                    cmin=float(min_count_abs_pct_error_mean),
                    cmax=float(max_count_abs_pct_error_mean),
                    size=np.where(scatter_df["is_glass_pair"], 13, 10),
                    opacity=0.78,
                    line=dict(
                        color=np.where(scatter_df["is_glass_pair"], "#000000", "#4b5563"),
                        width=np.where(scatter_df["is_glass_pair"], 3.2, 1.0),
                    ),
                    colorbar=dict(title="Mean abs count % error", len=0.7, thickness=18),
                ),
                customdata=scatter_df[
                    [
                        "pathogen",
                        "antibiotic",
                        "latest_available_year",
                        "future_year",
                        "latest_available_count",
                        "latest_smoothed_SI",
                        "future_predicted_SI",
                        "future_predicted_count",
                        "future_minus_latest_smoothed_SI",
                        "post_fit_count_abs_pct_error_mean",
                        "n_count_error_years",
                        "appears_in_GLASS",
                        "glass_group_column",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "<b>%{customdata[1]}</b><br>"
                    "Years: %{customdata[2]} -> %{customdata[3]}<br>"
                    "Counts: %{customdata[4]:.0f} -> %{customdata[7]:.0f}<br>"
                    "Smoothed SI: %{customdata[5]:.3f} -> %{customdata[6]:.3f}<br>"
                    "Delta SI: %{customdata[8]:.3f}<br>"
                    "Error (pc): %{customdata[9]:.2f}<br>"
                    "Count-error years: %{customdata[10]:.0f}<br>"
                    "GLASS: %{customdata[11]}<extra></extra>"
                ),
                name="Pairs",
            )
        ]
    )

    fig.update_layout(
        title=title,
        dragmode="pan",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        width=int(width),
        height=int(height),
        margin=dict(l=100, r=150, t=90, b=80),
    )
    if add_zero_lines:
        fig.add_hline(y=0.0, line_dash="dot", line_color="#111827", line_width=2, opacity=0.9)
        fig.add_vline(x=0.0, line_dash="dot", line_color="#111827", line_width=2, opacity=0.9)
    if add_identity_line:
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            xref="x domain",
            yref="y domain",
            line=dict(color="#111827", dash="dot", width=2),
            opacity=0.9,
        )
    return fig, scatter_df


def plot_future_si_vs_si_shift(
    plot_df: pd.DataFrame,
    *,
    min_count_abs_pct_error_mean: float = 0.0,
    max_count_abs_pct_error_mean: float = 10.0,
    display_name: bool = False,
    max_text_labels: int | None = None,
    always_label_glass: bool = True,
    title: str = "Predicted future SI vs predicted SI shift",
    width: int = 1400,
    height: int = 900,
    show: bool = True,
    html_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
) -> dict[str, Any]:
    """Plot predicted future SI against future-minus-latest smoothed SI shift."""
    required_cols = {
        "pathogen",
        "antibiotic",
        "latest_available_year",
        "future_year",
        "latest_available_count",
        "latest_smoothed_SI",
        "future_predicted_SI",
        "future_predicted_count",
        "future_minus_latest_smoothed_SI",
        "post_fit_count_abs_pct_error_mean",
        "n_count_error_years",
        "is_glass_pair",
        "glass_group_column",
    }
    missing = required_cols - set(plot_df.columns)
    if missing:
        raise KeyError(f"plot_df is missing required columns: {sorted(missing)}")

    try:
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover - optional notebook dependency
        raise ImportError("plotly is required for interactive SI plots.") from exc

    fig, scatter_df = _build_glass_si_scatter(
        plot_df,
        x_col="future_predicted_SI",
        y_col="future_minus_latest_smoothed_SI",
        title=title,
        xaxis_title="Predicted future SI",
        yaxis_title="Predicted future SI - smoothed SI at latest available year",
        min_count_abs_pct_error_mean=min_count_abs_pct_error_mean,
        max_count_abs_pct_error_mean=max_count_abs_pct_error_mean,
        display_name=display_name,
        max_text_labels=max_text_labels,
        always_label_glass=always_label_glass,
        width=width,
        height=height,
        add_zero_lines=True,
        add_identity_line=False,
    )

    saved_pdf_path, pdf_error, saved_html_path = _write_plotly_outputs(
        fig, html_path=html_path, pdf_path=pdf_path, width=width, height=height
    )
    if show:
        pio.renderers.default = "notebook_connected"
        fig.show()
    return {"figure": fig, "plot_df": scatter_df, "pdf_path": saved_pdf_path, "pdf_error": pdf_error, "html_path": saved_html_path}


def plot_latest_si_vs_future_si(
    plot_df: pd.DataFrame,
    *,
    min_count_abs_pct_error_mean: float = 0.0,
    max_count_abs_pct_error_mean: float = 10.0,
    display_name: bool = False,
    max_text_labels: int | None = None,
    always_label_glass: bool = True,
    title: str = "Latest available SI vs predicted future SI",
    width: int = 1400,
    height: int = 900,
    show: bool = True,
    html_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
) -> dict[str, Any]:
    """Plot latest available smoothed SI against predicted future SI."""
    required_cols = {
        "pathogen",
        "antibiotic",
        "latest_available_year",
        "future_year",
        "latest_available_count",
        "latest_smoothed_SI",
        "future_predicted_SI",
        "future_predicted_count",
        "future_minus_latest_smoothed_SI",
        "post_fit_count_abs_pct_error_mean",
        "n_count_error_years",
        "is_glass_pair",
        "glass_group_column",
    }
    missing = required_cols - set(plot_df.columns)
    if missing:
        raise KeyError(f"plot_df is missing required columns: {sorted(missing)}")

    try:
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover - optional notebook dependency
        raise ImportError("plotly is required for interactive SI plots.") from exc

    fig, scatter_df = _build_glass_si_scatter(
        plot_df,
        x_col="latest_smoothed_SI",
        y_col="future_predicted_SI",
        title=title,
        xaxis_title="Smoothed SI at latest available year",
        yaxis_title="Predicted future SI",
        min_count_abs_pct_error_mean=min_count_abs_pct_error_mean,
        max_count_abs_pct_error_mean=max_count_abs_pct_error_mean,
        display_name=display_name,
        max_text_labels=max_text_labels,
        always_label_glass=always_label_glass,
        width=width,
        height=height,
        add_zero_lines=True,
        add_identity_line=True,
    )

    saved_pdf_path, pdf_error, saved_html_path = _write_plotly_outputs(
        fig, html_path=html_path, pdf_path=pdf_path, width=width, height=height
    )
    if show:
        pio.renderers.default = "notebook_connected"
        fig.show()
    return {"figure": fig, "plot_df": scatter_df, "pdf_path": saved_pdf_path, "pdf_error": pdf_error, "html_path": saved_html_path}


__all__ = [
    "GLASS_MATCH_COLUMNS",
    "NOVEL_PREDICTION_COLUMNS",
    "REQUIRED_INTERACTIVE_RANK_COLUMNS",
    "build_glass_match_table",
    "build_novel_prediction_table",
    "load_or_build_interactive_rank",
    "load_si_and_clean_glass_list",
    "plot_future_si_vs_si_shift",
    "plot_latest_si_vs_future_si",
]
