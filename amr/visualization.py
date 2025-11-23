"""Visualization helpers for AMR analysis notebooks."""

from __future__ import annotations

import ast
import math
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, TwoSlopeNorm
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from umap import UMAP
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------
def plot_heatmap_for_pathogen(
    df: pd.DataFrame,
    pathogen: str,
    class_df: pd.DataFrame | None = None,
    group_column: str = "class",
    log_scale: bool = True,
    figsize: tuple[int, int] = (12, 6),
    show_only_class: bool = False,
):
    """Plot a heatmap of PMID counts for a given pathogen."""
    sub = df[df["pathogen"].str.lower() == pathogen.lower()].copy()

    if sub.empty:
        print(f"No data for pathogen: {pathogen}")
        return

    sub["year_mid"] = ((pd.to_datetime(sub["start_date"]).dt.year + pd.to_datetime(sub["end_date"]).dt.year) // 2) + 1

    group_column_normalized = group_column.strip().lower() if isinstance(group_column, str) and group_column.strip() else None
    group_label = group_column.strip() if isinstance(group_column, str) and group_column.strip() else "class"
    group_value_col = "_group_value"

    if class_df is not None:
        mapper = class_df.rename(columns={c: c.lower() for c in class_df.columns}).copy()
        if "antibiotic" not in mapper.columns:
            raise ValueError("class_df must contain an 'antibiotic' column")

        sub["antibiotic"] = sub["antibiotic"].str.lower()
        mapper["antibiotic"] = mapper["antibiotic"].str.lower()
        sub = sub.merge(mapper, on="antibiotic", how="left")

        if group_column_normalized is None:
            group_column_normalized = "class"
            group_label = "class"

        if group_column_normalized not in sub.columns:
            available = ", ".join(sorted(mapper.columns))
            raise ValueError(
                f"Column '{group_column}' not found in class_df. Available columns: {available}"
            )

        sub[group_value_col] = sub[group_column_normalized].fillna(sub["antibiotic"])
    else:
        if group_column_normalized not in (None, "class"):
            print(
                f"Grouping column '{group_column}' ignored because class_df was not provided; "
                "using antibiotic names instead."
            )
        group_label = "antibiotic"
        sub[group_value_col] = sub["antibiotic"]

    if show_only_class:
        sub = sub.groupby([group_value_col, "year_mid"], as_index=False)["count"].sum()
        index_col = group_value_col
    else:
        sub["antibiotic_prefixed"] = sub[group_value_col].astype(str) + " | " + sub["antibiotic"]
        sub = sub.groupby(["antibiotic_prefixed", "year_mid"], as_index=False)["count"].sum()
        index_col = "antibiotic_prefixed"

    pivot = sub.pivot(index=index_col, columns="year_mid", values="count").fillna(0)

    if pivot.empty:
        print("No data available for plotting.")
        return

    row_order = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]

    vmax = pivot.values.max()
    if vmax <= 0:
        print("All counts are zero for the selected pathogen.")
        return

    norm = LogNorm(vmin=1, vmax=vmax) if log_scale else None
    ylabel = group_label if show_only_class else f"Antibiotic (prefixed by {group_label})"

    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        norm=norm,
        cbar_kws={"label": "PMID count (log scale)" if log_scale else "PMID count"},
    )
    plt.title(f"PubMed resistance reports for {pathogen}")
    plt.xlabel("Midpoint of interval")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def _resolve_interval_column(df: pd.DataFrame):
    """Ensure an interval column is present and return metadata for plotting."""
    interval_candidates = [
        ("interval_label", "Interval"),
        ("interval", "Interval"),
        ("time_interval", "Time interval"),
        ("time_period", "Time period"),
        ("period", "Period"),
        ("year_interval", "Year interval"),
        ("year_bin", "Year bin"),
        ("decade", "Decade"),
    ]
    for column, axis_label in interval_candidates:
        if column in df.columns:
            return df, column, axis_label

    if {"start_date", "end_date"}.issubset(df.columns):
        df = df.copy()
        start_year = pd.to_datetime(df["start_date"]).dt.year.astype("Int64")
        end_year = pd.to_datetime(df["end_date"]).dt.year.astype("Int64")
        df["_interval_label"] = start_year.astype(str) + "-" + end_year.astype(str)
        return df, "_interval_label", "Interval"

    available = ", ".join(sorted(df.columns))
    candidate_names = [col for col, _ in interval_candidates]
    raise ValueError(
        "Unable to infer interval column; expected one of "
        f"{candidate_names} or 'start_date'/'end_date'. Available columns: {available}"
    )


def plot_heatmap_for_pathogen_debiased(
    df: pd.DataFrame,
    pathogen: str,
    class_df: pd.DataFrame | None = None,
    value: str = "zscore",
    figsize: tuple[int, int] = (12, 6),
    show_only_class: bool = False,
    vmin=None,
    vmax=None,
):
    """Plot a debiased heatmap that handles NaN/inf robustly."""
    sub = df[df["pathogen"].str.lower() == pathogen.lower()].copy()
    if sub.empty:
        print(f"No data for pathogen: {pathogen}")
        return

    if value not in sub.columns:
        raise ValueError(f"Column '{value}' not found in dataframe.")

    if class_df is not None:
        class_df = class_df.rename(columns={c: c.lower() for c in class_df.columns})
        sub["antibiotic"] = sub["antibiotic"].str.lower()
        class_df["antibiotic"] = class_df["antibiotic"].str.lower()
        sub = sub.merge(class_df, on="antibiotic", how="left")
        sub["class"] = sub["class"].fillna(sub["antibiotic"])
        sub, interval_col, x_label = _resolve_interval_column(sub)
        if show_only_class:
            sub = sub.groupby(["class", interval_col], as_index=False)[value].mean()
            index_col = "class"
        else:
            sub["antibiotic_prefixed"] = sub["class"] + " | " + sub["antibiotic"]
            sub = sub.groupby(["antibiotic_prefixed", interval_col], as_index=False)[value].mean()
            index_col = "antibiotic_prefixed"
    else:
        sub, interval_col, x_label = _resolve_interval_column(sub)
        index_col = "antibiotic"

    sub[value] = pd.to_numeric(sub[value], errors="coerce").replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=[value])

    pivot = sub.pivot(index=index_col, columns=interval_col, values=value)
    pivot = pivot.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pivot.index = pivot.index.astype(str)
    pivot.columns = [str(c) for c in pivot.columns]

    if not np.all(np.isfinite(pivot.values)):
        print("⚠️ Non-finite entries detected, replacing with 0.")
        pivot = pd.DataFrame(np.nan_to_num(pivot.values), index=pivot.index, columns=pivot.columns)

    if isinstance(pivot, pd.DataFrame):
        pivot = pivot.loc[pivot.sum(axis=1) != 0]
    if pivot.empty:
        print("No finite values left to plot.")
        return

    if isinstance(pivot, pd.DataFrame):
        row_order = pivot.abs().sum(axis=1).sort_values(ascending=False).index
        pivot = pivot.loc[row_order]

    vmax_auto = float(np.nanmax(np.abs(pivot.values)))
    if not np.isfinite(vmax_auto) or vmax_auto == 0:
        print("No finite range to plot.")
        return
    if vmin is None or vmax is None:
        vmin, vmax = -vmax_auto, vmax_auto
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    if not np.all(np.isfinite(pivot.values)):
        print("Non-finite data remain, skipping plot.")
        return
    if any(pd.isna(pivot.columns)) or any(pd.isna(pivot.index)):
        print("Non-finite tick labels, skipping plot.")
        return

    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        norm=norm,
        cbar_kws={"label": f"{value} (detrended)"},
        linewidths=0.3,
    )
    plt.title(f"Debiased {value} heatmap for {pathogen}")
    plt.xlabel(x_label)
    plt.ylabel("Antibiotic class" if show_only_class else "Antibiotic (prefixed by class)")
    plt.tight_layout()
    plt.show()


def _format_count_annotation(count_mat: pd.DataFrame, expected_mat: pd.DataFrame):
    observed = np.round(count_mat).astype(int).astype(str)
    expected = np.round(expected_mat).astype(int).astype(str)
    return observed + ":" + expected


def _plot_diff_heatmap(sub: pd.DataFrame, index_col: str, interval_col: str, x_label: str, figsize, vmin, vmax, title):
    diff_mat = sub.pivot(index=index_col, columns=interval_col, values="diff").fillna(0.0)
    count_mat = sub.pivot(index=index_col, columns=interval_col, values="count").fillna(0.0)
    expected_mat = sub.pivot(index=index_col, columns=interval_col, values="expected_count").fillna(0.0)

    if diff_mat.empty:
        print("No data available for the requested selection.")
        return

    max_abs = np.nanmax(np.abs(diff_mat.values))
    if not np.isfinite(max_abs) or max_abs == 0:
        print("No variation in residuals to plot.")
        return

    if vmin is None or vmax is None:
        vmin, vmax = -max_abs, max_abs

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    annot = _format_count_annotation(count_mat, expected_mat)

    plt.figure(figsize=figsize)
    sns.heatmap(
        diff_mat,
        cmap="RdBu_r",
        norm=norm,
        cbar_kws={"label": "log residual (observed - expected)"},
        annot=annot,
        fmt="",
        linewidths=0.3,
        linecolor="grey",
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.show()


def plot_heatmap_counts_vs_expected_antibiotic(
    df: pd.DataFrame,
    pathogen: str,
    class_df: pd.DataFrame | None = None,
    figsize: tuple[int, int] = (12, 6),
    vmin=None,
    vmax=None,
):
    """Plot pathogen vs antibiotic heatmap using marginal debiasing residuals."""
    sub = df[df["pathogen"].str.lower() == pathogen.lower()].copy()
    if sub.empty:
        print(f"No data for pathogen: {pathogen}")
        return

    if class_df is not None:
        mapper = class_df.rename(columns={c: c.lower() for c in class_df.columns}).copy()
        mapper["antibiotic"] = mapper["antibiotic"].str.lower()
        sub["antibiotic"] = sub["antibiotic"].str.lower()
        sub = sub.merge(mapper, on="antibiotic", how="left")
        sub["class"] = sub["class"].fillna(sub["antibiotic"])
        sub["antibiotic_prefixed"] = sub["class"] + " | " + sub["antibiotic"]
        index_col = "antibiotic_prefixed"
    else:
        index_col = "antibiotic"

    sub, interval_col, x_label = _resolve_interval_column(sub)
    title = f"{pathogen}: observed vs expected counts per {x_label.lower()} (antibiotic-level)"
    _plot_diff_heatmap(sub, index_col, interval_col, x_label, figsize, vmin, vmax, title)


def plot_heatmap_counts_vs_expected_class(
    df: pd.DataFrame,
    pathogen: str,
    figsize: tuple[int, int] = (10, 5),
    vmin=None,
    vmax=None,
    class_column: str = "class",
    class_df: pd.DataFrame | None = None,
):
    """Plot pathogen vs group heatmap using marginal debiasing residuals."""
    sub = df[df["pathogen"].str.lower() == pathogen.lower()].copy()
    if sub.empty:
        print(f"No data for pathogen: {pathogen}")
        return

    if not isinstance(class_column, str) or not class_column.strip():
        raise ValueError("class_column must be a non-empty string.")

    requested_key = class_column.strip().lower()
    sub_lookup = {col.lower(): col for col in sub.columns}
    resolved_col = sub_lookup.get(requested_key)

    if resolved_col is None:
        if class_df is None:
            available = ", ".join(sorted(sub.columns))
            raise ValueError(f"Column '{class_column}' not found in dataframe. Available columns: {available}")

        mapper = class_df.rename(columns={c: c.lower() for c in class_df.columns}).copy()
        if "class" not in mapper.columns:
            raise ValueError("class_df must contain a 'Class' column for mapping.")
        mapper["class"] = mapper["class"].str.lower()

        mapper_lookup = {col.lower(): col for col in mapper.columns}
        mapped_col = mapper_lookup.get(requested_key)
        if mapped_col is None:
            available = ", ".join(sorted(class_df.columns))
            raise ValueError(f"Column '{class_column}' not found in class_df. Available columns: {available}")

        mapping_series = (
            mapper[["class", mapped_col]]
            .dropna(subset=[mapped_col])
            .drop_duplicates(subset=["class"])
            .set_index("class")[mapped_col]
        )
        target_col = class_column if class_column not in sub.columns else f"{class_column}_mapped"
        match_key = sub["class"].str.lower()
        sub[target_col] = match_key.map(mapping_series).fillna(sub["class"])
        resolved_col = target_col
    else:
        target_col = resolved_col

    sub, interval_col, x_label = _resolve_interval_column(sub)

    group_cols = ["pathogen", resolved_col, interval_col]
    if not set(["count", "expected_count"]).issubset(sub.columns):
        missing = sorted({"count", "expected_count"} - set(sub.columns))
        raise ValueError(f"Dataframe is missing required columns for aggregation: {missing}")
    aggregated = sub.groupby(group_cols, as_index=False)[["count", "expected_count"]].sum()
    aggregated["log_count"] = np.log1p(aggregated["count"])
    aggregated["expected_log"] = np.log1p(aggregated["expected_count"])
    aggregated["diff"] = aggregated["log_count"] - aggregated["expected_log"]
    sub = aggregated
    friendly_group = class_column.replace("_", " ").strip() or class_column
    title = f"{pathogen}: observed vs expected counts per {x_label.lower()} ({friendly_group} level)"
    _plot_diff_heatmap(sub, resolved_col, interval_col, x_label, figsize, vmin, vmax, title)


# ---------------------------------------------------------------------
# Embedding and UMAP visuals
# ---------------------------------------------------------------------
def _coerce_embedding_vector(value):
    """Convert stored embedding representations into 1D float arrays."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.ravel()
    raise ValueError("Embeddings must be 1D sequences of numbers.")


def plot_umap_antibiotic_activity(
    data_df: pd.DataFrame,
    amr_df: pd.DataFrame,
    antibiotic: str,
    date,
    embedding_column: str = "anonymised_embedding",
    cmap: str = "viridis",
    size_column: str = "pathogenic_index",
    umap_kwargs: Optional[dict] = None,
    min_n_causation: int = 3,
    max_marker_size: int = 10,
    min_marker_size: int = 1,
    mp4_path: str | None = None,
    video_total_duration: float | None = None,
    video_ping_pong: bool = False,
    video_loop: bool = True,
    image_size: tuple[int, int] | None = (704, 512),
):
    """Project pathogen embeddings with UMAP and color by antibiotic activity."""
    from io import BytesIO

    if embedding_column not in data_df.columns:
        raise ValueError(f"Column '{embedding_column}' not found in data_df.")
    if size_column not in data_df.columns:
        raise ValueError(f"Column '{size_column}' not found in data_df.")
    if "n_causation" not in data_df.columns:
        raise ValueError("data_df must contain 'n_causation' to apply the filter.")

    working = data_df[data_df["n_causation"] >= min_n_causation].copy()
    if working.empty:
        raise ValueError("No species meet the min_n_causation threshold.")

    embeddings = working[embedding_column].apply(_coerce_embedding_vector)
    lengths = embeddings.map(len)
    if lengths.nunique() != 1:
        raise ValueError("Embeddings must all have the same dimensionality.")
    embedding_matrix = np.vstack(embeddings.to_list())

    reducer = UMAP(n_components=2, random_state=42, **(umap_kwargs or {}))
    coords = reducer.fit_transform(embedding_matrix)
    working["umap_x"] = coords[:, 0]
    working["umap_y"] = coords[:, 1]

    size_values = pd.to_numeric(working[size_column], errors="coerce")
    if size_values.isna().all():
        raise ValueError(f"Size column '{size_column}' does not contain numeric data.")
    size_values = size_values.fillna(size_values.median())
    if size_values.le(0).all():
        size_values = size_values - size_values.min() + 1
    working["size_value"] = size_values
    size_norm = size_values - size_values.min()
    if size_norm.max() > 0:
        size_norm = size_norm / size_norm.max()
    working["_size_for_plot"] = 0.2 + size_norm
    marker_pixel_sizes = min_marker_size + working["_size_for_plot"] * max_marker_size

    if isinstance(date, (list, tuple, set, np.ndarray, pd.Series)):
        date_values = list(date)
    else:
        date_values = [date]
    if not date_values:
        raise ValueError("date must be a non-empty string or list of strings")
    target_dates = pd.to_datetime(date_values)
    date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in date_values]

    antibiotic_norm = antibiotic.strip().lower()

    subset = amr_df.copy()
    subset["antibiotic_norm"] = subset["antibiotic"].str.strip().str.lower()
    subset = subset[subset["antibiotic_norm"] == antibiotic_norm]

    if not subset.empty:
        subset["start_date"] = pd.to_datetime(subset["start_date"])
        subset["end_date"] = pd.to_datetime(subset["end_date"])
        subset["log_count"] = np.log1p(subset["count"])
    else:
        print(f"No records for antibiotic '{antibiotic}'.")

    species_key = working["CanonicalSpecies"].str.lower()

    color_arrays = []
    global_max = 0.0
    for target_date, label in zip(target_dates, date_labels):
        color_series = pd.Series(0.0, index=working.index)
        if not subset.empty:
            mask = (subset["start_date"] <= target_date) & (subset["end_date"] >= target_date)
            interval = subset[mask].groupby("pathogen")["log_count"].sum()
            interval.index = interval.index.str.lower()
            color_series = species_key.map(interval).fillna(0.0)
        color_arrays.append((color_series, label))
        if not color_series.empty:
            global_max = max(global_max, float(color_series.max()))

    overall_max = global_max if global_max > 0 else 1.0

    # Optional MP4 animation export (preferred for reliable playback)
    if mp4_path is not None:
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("imageio is required to save animations. Install with `pip install imageio`.") from exc
        try:
            import kaleido  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("kaleido is required for figure export. Install with `pip install kaleido`.") from exc

        frames = []
        base_frame_count = len(color_arrays)
        for idx, (color_series, label) in enumerate(color_arrays):
            frac = 0.0 if base_frame_count <= 1 else idx / (base_frame_count - 1)
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=working["umap_x"],
                        y=working["umap_y"],
                        mode="markers",
                        marker=dict(
                            size=marker_pixel_sizes,
                            color=color_series,
                            cmin=0,
                            cmax=overall_max,
                            colorscale=cmap,
                            line=dict(color="black", width=0.6),
                        ),
                        text=working["CanonicalSpecies"],
                        hovertemplate="%{text}<extra></extra>",
                    ),
                ]
            )
            # progress bar using figure fraction coordinates (paper space)
            fig.update_layout(
                shapes=[
                    dict(
                        type="line",
                        xref="paper",
                        yref="paper",
                        x0=0,
                        y0=1.05,
                        x1=frac,
                        y1=1.05,
                        line=dict(color="black", width=4),
                    )
                ]
            )
            fig.update_layout(
                title=f"{antibiotic} @ {label}",
                xaxis=dict(visible=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False,
                width=image_size[0] if image_size else None,
                height=image_size[1] if image_size else None,
            )
            png_bytes = fig.to_image(format="png")
            frames.append(imageio.imread(BytesIO(png_bytes)))
        if video_ping_pong and len(frames) > 1:
            frames = frames + frames[-2:0:-1]  # forward then backward without repeating endpoints

        if video_total_duration and len(frames) > 0:
            frame_duration = video_total_duration / len(frames)
        else:
            frame_duration = 1.0 / max(len(frames), 1e-6)  # default 1s per frame if no duration given

        try:
            import imageio_ffmpeg  # noqa: F401
            with imageio.get_writer(mp4_path, fps=1.0 / max(frame_duration, 1e-6)) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"Saved MP4 animation to {mp4_path}")
        except Exception as exc:
            print(f"Could not save MP4 (install imageio-ffmpeg): {exc}")
        return working

    num_plots = len(color_arrays)
    cols = max(1, math.ceil(math.sqrt(num_plots)))
    rows = math.ceil(num_plots / cols)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"{antibiotic} @ {label}" for _, label in color_arrays])

    for idx, (color_series, label) in enumerate(color_arrays):
        row = idx // cols + 1
        col = idx % cols + 1
        fig.add_trace(
            go.Scatter(
                x=working["umap_x"],
                y=working["umap_y"],
                mode="markers",
                marker=dict(
                    size=marker_pixel_sizes,
                    color=color_series,
                    coloraxis="coloraxis",
                    line=dict(color="black", width=0.6),
                ),
                text=working["CanonicalSpecies"],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(visible=False, showgrid=False, zeroline=False, row=row, col=col)
        axis_index = idx + 1
        scale_anchor = "x" if axis_index == 1 else f"x{axis_index}"
        fig.update_yaxes(
            visible=False,
            showgrid=False,
            zeroline=False,
            scaleanchor=scale_anchor,
            scaleratio=1,
            row=row,
            col=col,
        )

    fig.update_layout(
        coloraxis=dict(colorscale=cmap, cmin=0, cmax=overall_max, colorbar=dict(title="log₁₊ count")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=500 * cols,
        height=500 * rows,
        showlegend=False,
    )
    fig.show()
    return working


def _extract_genus(names: Iterable[str]):
    return [name.split()[0] if isinstance(name, str) and name else "Unknown" for name in names]


def _resolve_color_sequence(color_map, n_categories: int):
    """Return a color list sized for the number of categories."""
    if isinstance(color_map, (list, tuple)):
        sequence = list(color_map)
    elif isinstance(color_map, str):
        sequence = None
        for palette_module in (px.colors.qualitative, px.colors.sequential, px.colors.diverging):
            palette = getattr(palette_module, color_map, None)
            if palette is not None:
                sequence = list(palette)
                break
        if sequence is None:
            from plotly.colors import sample_colorscale

            effective = max(1, n_categories)
            positions = np.linspace(0, 1, effective) if effective > 1 else [0.5]
            sequence = sample_colorscale(color_map, positions)
    else:
        raise ValueError("color_map must be a string or a sequence of colors.")

    if not sequence:
        raise ValueError(f"Unrecognized color_map '{color_map}'.")

    effective = max(1, n_categories)
    base_len = len(sequence)
    if base_len < effective:
        chunk_size = (effective + base_len - 1) // base_len
        expanded = []
        for color in sequence:
            expanded.extend([color] * chunk_size)
            if len(expanded) >= effective:
                break
        sequence = expanded[:effective]
    else:
        sequence = sequence[:effective]
    return sequence


def _umap_scatter(
    pathogen_names: Sequence[str],
    log_sums: Sequence[float],
    vectors: np.ndarray,
    title: str,
    marker_max_size: float = 50,
    figure_size: tuple[int, int] | int | None = (800, 600),
    color_map: str | Sequence[str] = "Rainbow",
    min_size: float = 0.0,
):
    """Create a 2D UMAP scatter plot with configurable presentation options."""
    vectors = np.asarray(vectors)
    if vectors.size == 0:
        raise ValueError("Input vectors are empty; compute pathogen vectors first.")
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array.")

    log_sums = np.asarray(log_sums, dtype=float)
    pathogen_names = np.asarray(pathogen_names)
    if log_sums.shape[0] != vectors.shape[0] or pathogen_names.shape[0] != vectors.shape[0]:
        raise ValueError("pathogen_names, log_sums, and vectors must align on the first dimension.")

    marker_max_size = float(marker_max_size)
    if marker_max_size <= 0:
        raise ValueError("marker_max_size must be greater than zero.")

    min_size = float(min_size)
    if min_size < 0:
        raise ValueError("min_size must be zero or positive.")

    mask = log_sums >= min_size
    if not np.any(mask):
        raise ValueError("No data points remain after applying min_size filter.")
    if not np.all(mask):
        log_sums = log_sums[mask]
        pathogen_names = pathogen_names[mask]
        vectors = vectors[mask]

    reducer = UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(vectors)

    pathogen_names = pathogen_names.tolist()
    genus = _extract_genus(pathogen_names)
    hover_names = pathogen_names

    marker_min_size = 1e-3
    marker_min_size = min(marker_min_size, marker_max_size)
    if log_sums.size > 0 and np.max(log_sums) > 0 and marker_max_size > marker_min_size:
        marker_sizes = ((log_sums / np.max(log_sums)) * (marker_max_size - marker_min_size)) + marker_min_size
    else:
        marker_sizes = np.full(log_sums.shape, marker_min_size, dtype=float)

    n_categories = len(set(genus))
    color_sequence = _resolve_color_sequence(color_map, n_categories)

    if figure_size is None:
        width, height = 600, 600
    elif isinstance(figure_size, (int, float)):
        width = height = int(figure_size)
    else:
        width, height = figure_size
        width, height = int(width), int(height)

    if width <= 0 or height <= 0:
        raise ValueError("figure_size dimensions must be positive.")

    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color=genus,
        size=marker_sizes.tolist(),
        hover_name=hover_names,
        title=title,
        color_discrete_sequence=color_sequence,
    )
    fig.update_layout(
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        legend_title="Genus",
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=width,
        height=height,
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="black"), sizemode="area"))
    return fig


__all__ = [
    "plot_heatmap_for_pathogen",
    "plot_heatmap_for_pathogen_debiased",
    "plot_heatmap_counts_vs_expected_antibiotic",
    "plot_heatmap_counts_vs_expected_class",
    "plot_umap_antibiotic_activity",
    "_umap_scatter",
]
