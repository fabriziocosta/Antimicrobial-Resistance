"""Helpers for the GLASS interactive scientific-interest notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from amr.plot_labels import format_pathogen_antibiotic_label

from amr.glass import load_and_clean_glass_list
from amr.si import rank_yearly_pairs_by_future_si_shift

REQUIRED_INTERACTIVE_RANK_COLUMNS = {
    "latest_available_count",
    "total_publications",
    "latest_available_predicted_SI",
    "post_fit_count_abs_pct_error_mean",
    "n_count_error_years",
    "curve_fit_end_year",
    "post_fit_validation_end_year",
    "si_loess_residual_variance_mean",
    "loess_frac",
    "loess_degree",
    "tail_damping_years",
    "min_amr_pair_publication_count",
    "min_num_years_with_valid_counts",
}

GLASS_MATCH_COLUMNS = [
    "pathogen",
    "antibiotic",
    "latest_available_year",
    "future_year",
    "latest_available_count",
    "total_publications",
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
    "total_publications",
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
    curve_fit_end_year: int | None = None,
    post_fit_validation_end_year: int | None = None,
    loess_frac: float = 0.6,
    loess_degree: int = 1,
    tail_damping_years: int | None = None,
    min_amr_pair_publication_count: float = 2.0,
    min_num_years_with_valid_counts: int | None = None,
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

    def _cached_value_matches(column: str, expected: int | float | None) -> bool:
        if rank_df.empty or column not in rank_df.columns:
            return False
        values = rank_df[column]
        if expected is None:
            return bool(values.isna().all())
        numeric_values = pd.to_numeric(values, errors="coerce").dropna().unique()
        return len(numeric_values) == 1 and float(numeric_values[0]) == float(expected)

    metadata_mismatch = False
    if not rank_df.empty:
        expected_curve_fit_end_year = fit_end_year if curve_fit_end_year is None else curve_fit_end_year
        if post_fit_validation_end_year is None:
            observed_years = pd.to_numeric(si_df["year"], errors="coerce").dropna().astype(int)
            complete_observed_years = observed_years.loc[observed_years < pd.Timestamp.today().year]
            expected_post_fit_validation_end_year = (
                fit_end_year if complete_observed_years.empty else int(complete_observed_years.max())
            )
        else:
            expected_post_fit_validation_end_year = post_fit_validation_end_year
        expected_minimum_valid_years = (
            3 if min_num_years_with_valid_counts is None else min_num_years_with_valid_counts
        )
        expected_tail_damping_years = (
            None if tail_damping_years is None else max(float(tail_damping_years), 1e-6)
        )
        metadata_mismatch = not (
            _cached_value_matches("fit_start_year", fit_start_year)
            and _cached_value_matches("fit_end_year", fit_end_year)
            and _cached_value_matches("curve_fit_end_year", expected_curve_fit_end_year)
            and _cached_value_matches("post_fit_validation_end_year", expected_post_fit_validation_end_year)
            and _cached_value_matches("num_future_years", num_future_years)
            and _cached_value_matches("loess_frac", loess_frac)
            and _cached_value_matches("loess_degree", loess_degree)
            and _cached_value_matches("tail_damping_years", expected_tail_damping_years)
            and _cached_value_matches(
                "min_amr_pair_publication_count", min_amr_pair_publication_count
            )
            and _cached_value_matches("min_num_years_with_valid_counts", expected_minimum_valid_years)
        )

    if rank_df.empty or missing_rank_columns or cache_is_stale or metadata_mismatch:
        if verbose and missing_rank_columns:
            print(f"Recomputing rank data because cached file is missing: {sorted(missing_rank_columns)}")
        if verbose and cache_is_stale:
            print(f"Recomputing rank data because {rank_output_path} is older than {source_path}")
        if verbose and metadata_mismatch:
            print("Recomputing rank data because cached fit metadata does not match the requested endpoints.")
        rank_df = rank_yearly_pairs_by_future_si_shift(
            si_df,
            fit_start_year=fit_start_year,
            fit_end_year=fit_end_year,
            curve_fit_end_year=curve_fit_end_year,
            post_fit_validation_end_year=post_fit_validation_end_year,
            num_future_years=num_future_years,
            save_path=str(rank_output_path),
            loess_frac=loess_frac,
            loess_degree=loess_degree,
            tail_damping_years=tail_damping_years,
            min_amr_pair_publication_count=min_amr_pair_publication_count,
            min_num_years_with_valid_counts=min_num_years_with_valid_counts,
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
        lambda row: format_pathogen_antibiotic_label(row["pathogen"], row["antibiotic"]),
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


def _suppress_occluded_plot_labels(
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    width: int,
    height: int,
    x_axis_range: tuple[float, float] | None,
    y_axis_range: tuple[float, float] | None,
) -> pd.DataFrame:
    """Keep labels whose estimated screen boxes clear other labels and markers."""
    plot_df = plot_df.copy()
    x_values = pd.to_numeric(plot_df[x_col], errors="coerce")
    y_values = pd.to_numeric(plot_df[y_col], errors="coerce")
    if x_axis_range is None:
        valid_x = x_values.dropna()
        x_min = float(valid_x.min()) if not valid_x.empty else 0.0
        x_max = float(valid_x.max()) if not valid_x.empty else 1.0
        x_pad = max(x_max - x_min, 1e-9) * 0.05
        x_axis_range = (x_min - x_pad, x_max + x_pad)
    if y_axis_range is None:
        valid_y = y_values.dropna()
        y_min = float(valid_y.min()) if not valid_y.empty else 0.0
        y_max = float(valid_y.max()) if not valid_y.empty else 1.0
        y_pad = max(y_max - y_min, 1e-9) * 0.05
        y_axis_range = (y_min - y_pad, y_max + y_pad)

    plot_width_px = max(int(width) - 250, 1)
    plot_height_px = max(int(height) - 170, 1)
    x_per_pixel = (x_axis_range[1] - x_axis_range[0]) / plot_width_px
    y_per_pixel = (y_axis_range[1] - y_axis_range[0]) / plot_height_px
    # The same approximate right-of-marker placement is used by both Plotly
    # text and the optional white label boxes below.
    marker_gap_px = 8
    point_x_px = (x_values - x_axis_range[0]) / x_per_pixel
    point_y_px = (y_values - y_axis_range[0]) / y_per_pixel

    candidate_mask = plot_df["display_label"].astype(str).str.len().gt(0)
    candidates = plot_df.loc[candidate_mask].copy()
    if candidates.empty:
        return plot_df
    # Prefer GLASS pairs, then the same high-shift/high-count candidates used
    # when selecting a capped set of labels.
    candidates = candidates.assign(
        _label_priority_glass=candidates["is_glass_pair"].fillna(False).astype(bool),
    ).sort_values(
        ["_label_priority_glass", "future_minus_latest_smoothed_SI", "latest_available_count"],
        ascending=[False, False, False],
        kind="stable",
    )

    accepted_boxes: list[tuple[float, float, float, float]] = []
    accepted_indices = []
    for row_index, row in candidates.iterrows():
        label = str(row["display_label"])
        lines = label.split("<br>")
        label_width = max(max((len(line) for line in lines), default=1) * 4.7 + 10, 18)
        label_height = max(len(lines) * 10.5 + 6, 16)
        left = float(point_x_px.loc[row_index]) + marker_gap_px
        right = left + label_width
        center_y = float(point_y_px.loc[row_index])
        top = center_y - label_height / 2
        bottom = center_y + label_height / 2
        outside_plot = left < 0 or right > plot_width_px or top < 0 or bottom > plot_height_px
        overlaps_label = any(
            left < other_right and right > other_left and top < other_bottom and bottom > other_top
            for other_left, other_right, other_top, other_bottom in accepted_boxes
        )
        if not outside_plot and not overlaps_label:
            accepted_indices.append(row_index)
            accepted_boxes.append((left, right, top, bottom))

    plot_df.loc[~plot_df.index.isin(accepted_indices), "display_label"] = ""
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
    avoid_label_occlusion: bool,
    show_label_background: bool,
    width: int,
    height: int,
    add_zero_lines: bool,
    add_identity_line: bool,
    x_axis_range: tuple[float, float] | None = None,
    y_axis_range: tuple[float, float] | None = None,
) -> tuple[Any, pd.DataFrame]:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional notebook dependency
        raise ImportError("plotly is required for interactive SI plots.") from exc

    scatter_df = scatter_df.copy()
    # Plotly draws later points in a trace over earlier ones. Keep that order
    # deterministic so greater x values sit on top of smaller x values.
    scatter_df = scatter_df.sort_values(x_col, ascending=True, na_position="first", kind="stable")
    scatter_df = _add_plot_labels(
        scatter_df,
        display_name=display_name,
        max_text_labels=max_text_labels,
        always_label_glass=always_label_glass,
    )
    if display_name and avoid_label_occlusion:
        scatter_df = _suppress_occluded_plot_labels(
            scatter_df,
            x_col=x_col,
            y_col=y_col,
            width=width,
            height=height,
            x_axis_range=x_axis_range,
            y_axis_range=y_axis_range,
        )
    show_text = display_name and scatter_df["display_label"].astype(str).str.len().gt(0).any()

    marker_trace = go.Scatter(
        x=scatter_df[x_col],
        y=scatter_df[y_col],
        mode="markers+text" if show_text and not show_label_background else "markers",
        text=scatter_df["display_label"] if show_text and not show_label_background else None,
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
    fig = go.Figure()
    if show_text and show_label_background:
        label_rows = scatter_df.loc[scatter_df["display_label"].astype(str).str.len().gt(0)]
        x_values = pd.to_numeric(scatter_df[x_col], errors="coerce").dropna()
        y_values = pd.to_numeric(scatter_df[y_col], errors="coerce").dropna()
        if x_axis_range is None:
            x_min = float(x_values.min()) if not x_values.empty else 0.0
            x_max = float(x_values.max()) if not x_values.empty else 1.0
            x_pad = max(x_max - x_min, 1e-9) * 0.05
            x_axis_range = (x_min - x_pad, x_max + x_pad)
        if y_axis_range is None:
            y_min = float(y_values.min()) if not y_values.empty else 0.0
            y_max = float(y_values.max()) if not y_values.empty else 1.0
            y_pad = max(y_max - y_min, 1e-9) * 0.05
            y_axis_range = (y_min - y_pad, y_max + y_pad)
        plot_width_px = max(int(width) - 250, 1)
        plot_height_px = max(int(height) - 170, 1)
        x_per_pixel = (x_axis_range[1] - x_axis_range[0]) / plot_width_px
        y_per_pixel = (y_axis_range[1] - y_axis_range[0]) / plot_height_px
        marker_gap_px = 8  # marker radius plus a small tangent gap
        point_x = pd.to_numeric(label_rows[x_col], errors="coerce")
        box_x0 = point_x + marker_gap_px * x_per_pixel
        text_y = pd.to_numeric(label_rows[y_col], errors="coerce")
        text_values = label_rows["display_label"].astype(str)
        # Add each box and its left-aligned text in sequence, so later boxes
        # cover earlier labels where they overlap. The marker trace is last.
        for x0, y, label in zip(box_x0, text_y, text_values):
            lines = label.split("<br>")
            max_line_length = max((len(line) for line in lines), default=1)
            box_width_px = max_line_length * 4.7 + 10
            box_width = box_width_px * x_per_pixel
            box_height = max(len(lines) * 10.5 + 6, 16) * y_per_pixel
            box_left = float(x0)
            box_right = box_left + box_width
            fig.add_trace(
                go.Scatter(
                    x=[box_left, box_right, box_right, box_left, box_left],
                    y=[
                        float(y) - box_height / 2,
                        float(y) - box_height / 2,
                        float(y) + box_height / 2,
                        float(y) + box_height / 2,
                        float(y) - box_height / 2,
                    ],
                    mode="lines",
                    fill="toself",
                    fillcolor="white",
                    line=dict(color="#000000", width=0.5),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Label background",
                )
            )
            line_y_positions = [
                float(y) + ((len(lines) - 1) / 2 - line_i) * 10.5 * y_per_pixel
                for line_i in range(len(lines))
            ]
            fig.add_trace(
                go.Scatter(
                    x=[box_left + 4 * x_per_pixel] * len(lines),
                    y=line_y_positions,
                    mode="text",
                    text=lines,
                    textposition="middle right",
                    textfont=dict(size=8, color="#2a3f5f"),
                    hoverinfo="skip",
                    showlegend=False,
                    cliponaxis=False,
                    name="Pair label",
                )
            )
    fig.add_trace(marker_trace)

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
    predicted_future_si_threshold: float = 0.0,
    predicted_future_delta_si_threshold: float = 0.0,
    show_mean_count_pct_error: bool = False,
    display_name: bool = False,
    max_text_labels: int | None = None,
    always_label_glass: bool = True,
    avoid_label_occlusion: bool = True,
    show_label_background: bool = False,
    title: str = "Predicted future SI vs predicted SI shift",
    width: int = 1400,
    height: int = 900,
    show: bool = True,
    show_quadrants: bool | None = None,
    quadrant_output_dir: str | Path | None = None,
    html_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
) -> dict[str, Any]:
    """Plot future SI against SI shift with threshold-defined quadrants."""
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

    x_values_for_range = pd.to_numeric(plot_df["future_predicted_SI"], errors="coerce").dropna()
    y_values_for_range = pd.to_numeric(plot_df["future_minus_latest_smoothed_SI"], errors="coerce").dropna()
    x_span_for_range = max(float(x_values_for_range.max() - x_values_for_range.min()), 1.0) if not x_values_for_range.empty else 1.0
    y_span_for_range = max(float(y_values_for_range.max() - y_values_for_range.min()), 1.0) if not y_values_for_range.empty else 1.0
    x_min_for_range = (
        min(float(x_values_for_range.min()), float(predicted_future_si_threshold)) - 0.05 * x_span_for_range
        if not x_values_for_range.empty
        else float(predicted_future_si_threshold) - 1.0
    )
    x_max_for_range = (
        max(float(x_values_for_range.max()), float(predicted_future_si_threshold)) + 0.05 * x_span_for_range
        if not x_values_for_range.empty
        else float(predicted_future_si_threshold) + 1.0
    )
    y_min_for_range = (
        min(float(y_values_for_range.min()), float(predicted_future_delta_si_threshold)) - 0.05 * y_span_for_range
        if not y_values_for_range.empty
        else float(predicted_future_delta_si_threshold) - 1.0
    )
    y_max_for_range = (
        max(float(y_values_for_range.max()), float(predicted_future_delta_si_threshold)) + 0.05 * y_span_for_range
        if not y_values_for_range.empty
        else float(predicted_future_delta_si_threshold) + 1.0
    )

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
        avoid_label_occlusion=avoid_label_occlusion,
        show_label_background=show_label_background,
        width=width,
        height=height,
        x_axis_range=(x_min_for_range, x_max_for_range),
        y_axis_range=(y_min_for_range, y_max_for_range),
        add_zero_lines=False,
        add_identity_line=False,
    )
    fig.update_layout(
        font=dict(size=10),
        title_font=dict(size=14),
        xaxis_title_font=dict(size=11),
        yaxis_title_font=dict(size=11),
        xaxis_tickfont=dict(size=9),
        yaxis_tickfont=dict(size=9),
    )
    if fig.data and fig.data[0].textfont is not None:
        fig.data[0].textfont.size = 8

    # Shade the two below-threshold bands. Their overlap gets a blended shade,
    # while the quadrant above both thresholds remains white.
    x_values = pd.to_numeric(scatter_df["future_predicted_SI"], errors="coerce").dropna()
    y_values = pd.to_numeric(scatter_df["future_minus_latest_smoothed_SI"], errors="coerce").dropna()
    x_span = max(float(x_values.max() - x_values.min()), 1.0) if not x_values.empty else 1.0
    y_span = max(float(y_values.max() - y_values.min()), 1.0) if not y_values.empty else 1.0
    x_min = (
        min(float(x_values.min()), float(predicted_future_si_threshold)) - 0.05 * x_span
        if not x_values.empty
        else float(predicted_future_si_threshold) - 1.0
    )
    y_min = (
        min(float(y_values.min()), float(predicted_future_delta_si_threshold)) - 0.05 * y_span
        if not y_values.empty
        else float(predicted_future_delta_si_threshold) - 1.0
    )
    x_max = (
        max(float(x_values.max()), float(predicted_future_si_threshold)) + 0.05 * x_span
        if not x_values.empty
        else float(predicted_future_si_threshold) + 1.0
    )
    y_max = (
        max(float(y_values.max()), float(predicted_future_delta_si_threshold)) + 0.05 * y_span
        if not y_values.empty
        else float(predicted_future_delta_si_threshold) + 1.0
    )
    # Match the axis bounds to the shaded bands so Plotly autorange padding
    # cannot leave a white strip at the left or bottom edges.
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])
    fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=x_min,
        x1=float(predicted_future_si_threshold),
        y0=0,
        y1=1,
        fillcolor="#60a5fa",
        opacity=0.3,
        line_width=0,
        layer="below",
    )
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=y_min,
        y1=float(predicted_future_delta_si_threshold),
        fillcolor="#f59e0b",
        opacity=0.3,
        line_width=0,
        layer="below",
    )
    fig.add_vline(
        x=float(predicted_future_si_threshold),
        line_dash="dot",
        line_color="#111827",
        line_width=2,
        opacity=0.9,
    )
    fig.add_hline(
        y=float(predicted_future_delta_si_threshold),
        line_dash="dot",
        line_color="#111827",
        line_width=2,
        opacity=0.9,
    )

    if not show_mean_count_pct_error:
        # The shared scatter helper uses error as the fill color. Remove that
        # encoding when hollow markers are requested, and remove its colorbar.
        trace = fig.data[-1]
        trace.marker.symbol = "circle-open"
        trace.marker.color = np.where(scatter_df["is_glass_pair"], "#000000", "#4b5563")
        trace.marker.colorbar = None
        trace.marker.colorscale = None
        trace.marker.showscale = False

    saved_pdf_path, pdf_error, saved_html_path = _write_plotly_outputs(
        fig, html_path=html_path, pdf_path=pdf_path, width=width, height=height
    )
    if show:
        pio.renderers.default = "notebook_connected"
        fig.show()

    quadrant_results: dict[str, dict[str, Any]] = {}
    if show_quadrants is None:
        show_quadrants = show
    if quadrant_output_dir is not None:
        quadrant_output_dir = Path(quadrant_output_dir)
        quadrant_output_dir.mkdir(parents=True, exist_ok=True)

    x = pd.to_numeric(scatter_df["future_predicted_SI"], errors="coerce")
    y = pd.to_numeric(scatter_df["future_minus_latest_smoothed_SI"], errors="coerce")
    x_threshold = float(predicted_future_si_threshold)
    y_threshold = float(predicted_future_delta_si_threshold)
    quadrant_masks = {
        "below_both": x.lt(x_threshold) & y.lt(y_threshold),
        "below_future_si": x.lt(x_threshold) & y.ge(y_threshold),
        "below_delta_si": x.ge(x_threshold) & y.lt(y_threshold),
        "above_both": x.ge(x_threshold) & y.ge(y_threshold),
    }
    quadrant_titles = {
        "below_both": "Below both thresholds",
        "below_future_si": "Future SI below threshold; delta SI above",
        "below_delta_si": "Future SI above threshold; delta SI below",
        "above_both": "Above both thresholds",
    }
    for quadrant_name, mask in quadrant_masks.items():
        quadrant_df = scatter_df.loc[mask].copy()
        quadrant_fig, quadrant_plot_df = _build_glass_si_scatter(
            quadrant_df,
            x_col="future_predicted_SI",
            y_col="future_minus_latest_smoothed_SI",
            title=f"{quadrant_titles[quadrant_name]} (n={len(quadrant_df):,})",
            xaxis_title="Predicted future SI",
            yaxis_title="Predicted future SI - smoothed SI at latest available year",
            min_count_abs_pct_error_mean=min_count_abs_pct_error_mean,
            max_count_abs_pct_error_mean=max_count_abs_pct_error_mean,
            display_name=display_name,
            max_text_labels=max_text_labels,
            always_label_glass=always_label_glass,
            avoid_label_occlusion=avoid_label_occlusion,
            show_label_background=show_label_background,
            width=width,
            height=height,
            add_zero_lines=False,
            add_identity_line=False,
        )
        quadrant_fig.update_layout(
            font=dict(size=10),
            title_font=dict(size=14),
            xaxis_title_font=dict(size=11),
            yaxis_title_font=dict(size=11),
            xaxis_tickfont=dict(size=9),
            yaxis_tickfont=dict(size=9),
        )
        if quadrant_fig.data and quadrant_fig.data[0].textfont is not None:
            quadrant_fig.data[0].textfont.size = 8
        if not show_mean_count_pct_error and quadrant_fig.data:
            trace = quadrant_fig.data[-1]
            trace.marker.symbol = "circle-open"
            trace.marker.color = np.where(quadrant_plot_df["is_glass_pair"], "#000000", "#4b5563")
            trace.marker.colorbar = None
            trace.marker.colorscale = None
            trace.marker.showscale = False

        quadrant_pdf_path = None
        quadrant_pdf_error = None
        quadrant_html_path = None
        if quadrant_output_dir is not None:
            quadrant_pdf_path, quadrant_pdf_error, quadrant_html_path = _write_plotly_outputs(
                quadrant_fig,
                html_path=None,
                pdf_path=quadrant_output_dir / f"future_si_vs_si_shift_{quadrant_name}.pdf",
                width=width,
                height=height,
            )
        if show_quadrants:
            pio.renderers.default = "notebook_connected"
            quadrant_fig.show()
        quadrant_results[quadrant_name] = {
            "figure": quadrant_fig,
            "plot_df": quadrant_plot_df,
            "pdf_path": quadrant_pdf_path,
            "pdf_error": quadrant_pdf_error,
            "html_path": quadrant_html_path,
        }

    return {
        "figure": fig,
        "plot_df": scatter_df,
        "pdf_path": saved_pdf_path,
        "pdf_error": pdf_error,
        "html_path": saved_html_path,
        "quadrants": quadrant_results,
    }


def plot_latest_si_vs_future_si(
    plot_df: pd.DataFrame,
    *,
    min_count_abs_pct_error_mean: float = 0.0,
    max_count_abs_pct_error_mean: float = 10.0,
    display_name: bool = False,
    max_text_labels: int | None = None,
    always_label_glass: bool = True,
    avoid_label_occlusion: bool = True,
    show_label_background: bool = False,
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
        avoid_label_occlusion=avoid_label_occlusion,
        show_label_background=show_label_background,
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
