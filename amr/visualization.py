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
    sort_by_frequency: bool = False,
):
    """Plot a heatmap of PMID counts for a given pathogen.

    Parameters
    - sort_by_frequency: if True, sort rows by descending cumulative counts; if False, sort alphabetically.
    """
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

    if sort_by_frequency:
        row_order = pivot.sum(axis=1).sort_values(ascending=False).index
        pivot = pivot.loc[row_order]
    else:
        pivot = pivot.sort_index()

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


def plot_venue_vector(
    df_with_vecs: pd.DataFrame,
    venue_index: dict[str, int],
    row: Optional[int] = None,
    ax=None,
    rotation: int = 0,
    title: Optional[str] = None,
    use_all_dates: bool = False,
    pathogen: Optional[str] = None,
    antibiotic: Optional[str] = None,
):
    """Bar plot of venue counts for a given pathogen/antibiotic row."""
    required_cols = {"venue_vector", "pathogen", "antibiotic"}
    missing = required_cols - set(df_with_vecs.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")

    if not isinstance(venue_index, dict):
        raise TypeError("venue_index must be a dictionary of venue->position.")

    if row is not None and (row < 0 or row >= len(df_with_vecs)):
        raise IndexError("Row index out of range for venue_vector plot.")

    if row is None:
        if pathogen is None or antibiotic is None:
            raise ValueError("Provide either `row` or both `pathogen` and `antibiotic`.")
        row_mask = (df_with_vecs["pathogen"] == pathogen) & (df_with_vecs["antibiotic"] == antibiotic)
        if not row_mask.any():
            raise ValueError("No rows match the specified pathogen/antibiotic combination.")
        current_row = df_with_vecs.loc[row_mask].iloc[0]
    else:
        current_row = df_with_vecs.iloc[row]
        pathogen = pathogen or current_row["pathogen"]
        antibiotic = antibiotic or current_row["antibiotic"]

    venues_sorted = [venue for venue, _ in sorted(venue_index.items(), key=lambda kv: kv[1])]

    def _coerce_vector(vector_like):
        if not isinstance(vector_like, (list, tuple)):
            raise TypeError("venue_vector entries must be list-like.")
        vec = list(vector_like)
        if len(vec) != len(venues_sorted):
            adjusted = [0] * len(venues_sorted)
            for venue, idx in venue_index.items():
                if idx < len(vec):
                    adjusted[idx] = vec[idx]
            vec = adjusted
        return vec

    if use_all_dates:
        mask = (df_with_vecs["pathogen"] == pathogen) & (df_with_vecs["antibiotic"] == antibiotic)
        grouped = df_with_vecs.loc[mask, "venue_vector"]
        aggregated = [0] * len(venues_sorted)
        for vector_like in grouped:
            try:
                vec = _coerce_vector(vector_like)
            except TypeError:
                continue
            for idx, value in enumerate(vec):
                aggregated[idx] += value
        vector = aggregated
    else:
        vector = _coerce_vector(current_row["venue_vector"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, max(4, len(venues_sorted) * 0.35)))
    else:
        fig = ax.figure

    positions = list(range(len(venues_sorted)))
    max_value = max(vector) if vector else 0
    offset = 0.02 * max_value if max_value else 0.1
    bars = ax.barh(positions, vector)
    for bar, value in zip(bars, vector):
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height() / 2
        x_pos = width + offset if width else offset
        ax.text(x_pos, y_pos, f"{value}", va="center", ha="left", fontsize=8)
    ax.set_yticks(positions)
    ax.set_yticklabels(venues_sorted, rotation=rotation)
    ax.set_xlabel("Count")
    ax.set_ylabel("Venue")
    if title:
        ax.set_title(title)
    ax.margins(y=0.01)
    try:
        fig.tight_layout()
    except ValueError:
        pass
    return ax


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


def plot_top_antibiotic_period_by_pathogen(
    norm_df: pd.DataFrame,
    values: Sequence[str] = ("z_bg", "resid_log_bg"),
    top_k: int = 3,
    cmap: str = "Reds",
    color_by: str = "rank",
    rank_min_intensity: float = 0.25,
    data_df: pd.DataFrame | None = None,
    top_k_pathogens: int = 10,
    top_k_antibiotics: int = 20,
    pathogen_name_col: str = "CanonicalSpecies",
    pathogen_rank_col: str = "pathogenic_index",
    width_multiplier: float = 1.0,
    figsize: tuple[int, int] | None = None,
    annotate: bool = True,
    xtick_rotation: int = 45,
    invert: bool = False,
):
    """
    For each pathogen, find the period (interval) and antibiotic with the
    maximum signal across all antibiotics and periods, and plot them.

    Parameters
    - norm_df: DataFrame with columns: pathogen, antibiotic, start_date, end_date,
               interval label (or start/end to infer), and at least one of
               {"z_bg", "resid_log_bg"}.
    - values: one or more metric names to plot; defaults to ("z_bg", "resid_log_bg").
              Each metric produces a panel showing the peak per-pathogen.
    - data_df: optional DataFrame with pathogen rankings; expects columns
               pathogen_name_col (default CanonicalSpecies) and pathogen_rank_col
               (default pathogenic_index). If provided, pathogens are ordered by
               descending rank and limited to top_k_pathogens.
    - top_k_pathogens: number of pathogens to include based on ranking (default 10).
    - top_k_antibiotics: number of antibiotics to include when invert=True (default 20).
    - width_multiplier: scales the default figure width (ignored if figsize given).
    - annotate: whether to label points with the antibiotic name.
    - xtick_rotation: rotation for interval labels on the x-axis.
    - invert: if False (default), rows are pathogens and point labels are
              antibiotics. If True, rows are antibiotics and point labels are
              pathogens.

    The x-axis shows interval labels. By default, the y-axis lists pathogens
    and each point is annotated by antibiotic. With invert=True, the y-axis
    lists antibiotics and points are annotated by pathogen.

    Coloring
    - color_by="rank" (default): within each pathogen's top-k, the top entry is
      darkest and the k-th is lightest, independent of absolute value. Set
      rank_min_intensity (0..1) to control the lightest shade.
    - color_by="value": color encodes the actual metric magnitude.
    """
    if norm_df.empty:
        print("norm_df is empty; nothing to plot.")
        return

    df = norm_df.copy()
    # Ensure interval label exists
    df, interval_col, x_label = _resolve_interval_column(df)

    # Accept either a single metric string or a sequence of metric names
    if isinstance(values, str):
        values = [values]
    else:
        values = list(values)
    requested_values = list(values)

    # Coerce metrics to numeric and keep only finite values
    values = [v for v in values if v in df.columns]
    if not values:
        available = ", ".join(sorted(df.columns))
        raise ValueError(
            "None of the requested values are present in norm_df. "
            f"Requested: {requested_values}. Available columns: {available}"
        )
    for v in values:
        df[v] = pd.to_numeric(df[v], errors="coerce")

    # Order intervals by start year if available; otherwise alphabetical
    if {"start_date", "end_date"}.issubset(df.columns):
        df["_start_year"] = pd.to_datetime(df["start_date"]).dt.year.astype("Int64")
        df["_end_year"] = pd.to_datetime(df["end_date"]).dt.year.astype("Int64")
        interval_order = (
            df[[interval_col, "_start_year", "_end_year"]]
            .drop_duplicates(subset=[interval_col])
            .sort_values(by=["_start_year", "_end_year", interval_col])
        )
        intervals_sorted = interval_order[interval_col].astype(str).tolist()
    else:
        intervals_sorted = sorted(df[interval_col].astype(str).unique().tolist())

    # Build figure
    n_panels = len(values)
    if figsize is None:
        # Height scales with #pathogens; estimate after computing order
        pass

    # Determine row axis and labels (default: pathogen rows, inverted: antibiotic rows)
    row_col = "antibiotic" if invert else "pathogen"
    label_col = "pathogen" if invert else "antibiotic"
    row_axis_label = "Antibiotic" if invert else "Pathogen"
    row_title_suffix = "antibiotic" if invert else "pathogen"
    row_values_present = df[row_col].astype(str).unique().tolist()
    rows_order: list[str]
    if (
        not invert
        and data_df is not None
        and pathogen_name_col in data_df.columns
        and pathogen_rank_col in data_df.columns
    ):
        dd = data_df[[pathogen_name_col, pathogen_rank_col]].copy()
        dd["_name_l"] = dd[pathogen_name_col].astype(str).str.strip().str.lower()
        # Aggregate duplicates via max rank
        rank_series = dd.groupby("_name_l")[pathogen_rank_col].max().sort_values(ascending=False)
        pp = pd.DataFrame({"orig": pd.Series(row_values_present, dtype=str)})
        pp["_name_l"] = pp["orig"].str.strip().str.lower()
        pp["rank"] = pp["_name_l"].map(rank_series)
        pp = pp.sort_values(by=["rank"], ascending=[False], na_position="last")
        rows_order = pp["orig"].tolist()
    else:
        # Fallback: order rows by maximum of the first metric
        v0 = values[0]
        base_cols = [row_col, interval_col, v0]
        if "antibiotic" not in base_cols:
            base_cols.insert(1, "antibiotic")
        df0 = df[base_cols].copy()
        df0 = df0.replace([np.inf, -np.inf], np.nan).dropna(subset=[v0])
        if df0.empty:
            print(f"No finite values for {v0}; nothing to plot.")
            return
        per_row_max = df0.groupby(row_col)[v0].max().sort_values(ascending=False)
        rows_order = per_row_max.index.tolist()

    if invert:
        if isinstance(top_k_antibiotics, int) and top_k_antibiotics > 0:
            rows_order = rows_order[: top_k_antibiotics]
    else:
        if isinstance(top_k_pathogens, int) and top_k_pathogens > 0:
            rows_order = rows_order[: top_k_pathogens]

    # Default figsize after we know how many pathogens
    if figsize is None:
        height = max(6, int(len(rows_order) * 0.35) + 2)
        width = 10 if n_panels == 1 else 12 + 4 * (n_panels - 1)
        try:
            wm = float(width_multiplier)
            if wm > 0:
                width = int(round(width * wm))
        except Exception:
            pass
        figsize = (width, height)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes = axes[0]

    # Helper to plot a single panel
    def _panel(ax, metric: str, title_suffix: str):
        dfx = df[["pathogen", "antibiotic", interval_col, metric]].copy()
        dfx = dfx.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
        if dfx.empty:
            ax.set_visible(False)
            return
        # Take top-k rows per selected row entity (pathogen or antibiotic) by metric
        dfx_sorted = dfx.sort_values(by=metric, ascending=False)
        top = dfx_sorted.groupby(row_col).head(max(1, int(top_k)))
        # Align to common y order; drop rows not present in base order
        top = top[top[row_col].isin(rows_order)]
        # Compute color values
        if color_by.lower() == "rank":
            # Rank within each row's selected points (1=top)
            top = top.sort_values([row_col, metric], ascending=[True, False])
            top["_rank"] = top.groupby(row_col).cumcount() + 1
            # Normalize to [rank_min_intensity, 1.0]
            # If only one item, assign 1.0
            def _rank_to_intensity(gr):
                n = len(gr)
                if n <= 1:
                    return pd.Series([1.0] * n, index=gr.index)
                lo = float(rank_min_intensity)
                hi = 1.0
                # Highest rank (1) -> hi, lowest rank (n) -> lo
                scale = np.linspace(hi, lo, n)
                return pd.Series(scale, index=gr.index)

            top["_color_val"] = (
                top.groupby(row_col, group_keys=False)["_rank"].apply(lambda g: _rank_to_intensity(g))
            )
            c_vals = top["_color_val"].astype(float).values
            vmin, vmax = 0.0, 1.0
        else:
            c_vals = top[metric].astype(float).values
            vmin = float(np.nanmin(c_vals)) if c_vals.size else None
            vmax = float(np.nanmax(c_vals)) if c_vals.size else None
        # Map categories to positions
        y_map = {p: i for i, p in enumerate(rows_order)}
        x_map = {lab: i for i, lab in enumerate(intervals_sorted)}
        top = top.assign(
            x_idx=top[interval_col].astype(str).map(x_map),
            y_base=top[row_col].map(y_map).astype(float),
        )
        # If multiple top entries fall on the same interval for a pathogen,
        # spread them vertically to avoid overlap (stack as multiple lines)
        top["_yoff"] = 0.0
        groups = top.groupby([row_col, "x_idx"], sort=False)
        for (_, _), idxs in groups.groups.items():
            n = len(idxs)
            if n > 1:
                # symmetric offsets within the unit row height
                spread = min(0.4, 0.15 * (n - 1))
                offsets = np.linspace(-spread, spread, n)
                top.loc[list(idxs), "_yoff"] = offsets

        xs = top["x_idx"].values
        ys = (top["y_base"] + top["_yoff"]).values
        # Alternating horizontal bands per pathogen row to improve readability
        try:
            nrows = len(rows_order)
            for i in range(nrows):
                if i % 2 == 0:
                    ax.axhspan(i - 0.5, i + 0.5, color="#f0f0f0", zorder=0)
        except Exception:
            pass
        sc = ax.scatter(xs, ys, c=c_vals, cmap=cmap, edgecolor="k", s=60, vmin=vmin, vmax=vmax)
        if annotate:
            for (x, y, label) in zip(xs, ys, top[label_col]):
                ax.text(x + 0.1, y, str(label), va="center", ha="left", fontsize=8)

        ax.set_yticks(range(len(rows_order)))
        ax.set_yticklabels(rows_order)
        # Mirror y-axis labels on the right for easier scanning
        ax.yaxis.set_ticks_position("both")
        ax.tick_params(axis="y", which="both", left=True, right=True, labelleft=True, labelright=True)
        # Reserve one empty column on the right to avoid clipping annotations
        ncols = len(intervals_sorted)
        ax.set_xticks(range(ncols + 1))
        ax.set_xticklabels(list(intervals_sorted) + [""], rotation=xtick_rotation, ha="right")
        ax.set_xlim(-0.5, ncols - 0.5 + 1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(row_axis_label)
        ax.set_title(f"Top {top_k} {title_suffix} per {row_title_suffix}")
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    for i, metric in enumerate(values):
        suffix = metric
        _panel(axes[i], metric, suffix)

    # Improve layout
    plt.tight_layout()
    plt.show()


def plot_top_pairs_by_period(
    norm_df: pd.DataFrame,
    value_col: str = "resid_log_bg",
    top_k: int = 10,
    period_col: str | None = None,
    stack_labels: bool = True,
    use_rank_fallback: bool = False,
    min_gap_ratio: float = 0.03,
    figsize: tuple[int, int] = (16, 9),
    width_multiplier: float = 1.0,
    fontsize: int = 8,
    label_x_base_offset: float = 0.10,
    label_x_step: float = 0.02,
    label_x_pad: float = 0.01,
    connector_dx_space: float = 0.0,
):
    """
    Plot top-k (antibiotic, pathogen) pairs per period as text labels.

    - x-axis: period
    - y-axis: value_col (default resid_log_bg)
    - text label: "(antibiotic, pathogen)"

    Readability options:
    - stack_labels=True: vertically separate labels in each period when
      values are too close.
    - use_rank_fallback=True: use rank (1=highest) as y-axis instead of value.

    Returns:
    - DataFrame with plotted rows and helper coordinates.
    """
    result_cols = [
        "antibiotic",
        "pathogen",
        "period",
        "pair_label",
        "rank",
        "value_raw",
        "y_raw",
        "y_plot",
        "x_idx",
        "x_marker",
        "x_conn",
        "x_text",
    ]
    if norm_df is None or norm_df.empty:
        print("norm_df is empty; nothing to plot.")
        return pd.DataFrame(columns=result_cols)

    required_cols = {"antibiotic", "pathogen", value_col}
    missing = required_cols - set(norm_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = norm_df.copy()

    if period_col is None:
        for col in [
            "interval_label",
            "interval",
            "time_interval",
            "period",
            "year_bin",
            "decade",
        ]:
            if col in df.columns:
                period_col = col
                break

    if period_col is None and {"start_date", "end_date"}.issubset(df.columns):
        start_year = pd.to_datetime(df["start_date"], errors="coerce").dt.year.astype("Int64")
        end_year = pd.to_datetime(df["end_date"], errors="coerce").dt.year.astype("Int64")
        df["_period_label"] = start_year.astype(str) + "-" + end_year.astype(str)
        period_col = "_period_label"

    if period_col is None or period_col not in df.columns:
        raise ValueError("Could not infer period column. Provide `period_col` explicitly.")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    # Avoid DataFrame-wide replace here because some notebooks keep array-like
    # objects in other columns (e.g., embeddings), which can break comparisons.
    df = df[np.isfinite(df[value_col])]
    df = df.dropna(subset=["antibiotic", "pathogen", period_col])
    if df.empty:
        print(f"No finite {value_col} values after filtering.")
        return pd.DataFrame(columns=result_cols)

    top = (
        df.sort_values([period_col, value_col], ascending=[True, False])
        .groupby(period_col, group_keys=False)
        .head(max(1, int(top_k)))
        .copy()
    )
    top["pair_label"] = top["antibiotic"].astype(str) + "\n" + top["pathogen"].astype(str)
    top["_rank"] = top.groupby(period_col).cumcount() + 1

    if {"start_date", "end_date"}.issubset(df.columns):
        order_df = df[[period_col, "start_date", "end_date"]].drop_duplicates().copy()
        order_df["_s"] = pd.to_datetime(order_df["start_date"], errors="coerce")
        order_df["_e"] = pd.to_datetime(order_df["end_date"], errors="coerce")
        order_df = order_df.sort_values(["_s", "_e", period_col])
        periods = order_df[period_col].astype(str).tolist()
    else:
        periods = sorted(top[period_col].astype(str).unique().tolist())

    period_map = {p: i for i, p in enumerate(periods)}
    top["_x"] = top[period_col].astype(str).map(period_map)

    if use_rank_fallback:
        top["_y_raw"] = top["_rank"].astype(float)
        top["_y"] = top["_y_raw"]
        y_label = f"Rank by {value_col} (1=highest)"
    else:
        top["_y_raw"] = top[value_col].astype(float)
        top["_y"] = top["_y_raw"]
        y_label = value_col

        if stack_labels:
            y_min = float(top["_y_raw"].min())
            y_max = float(top["_y_raw"].max())
            y_span = max(y_max - y_min, 1e-9)
            min_gap = max(y_span * float(min_gap_ratio), 1e-6)

            for _, grp in top.groupby(period_col, sort=False):
                idx = grp.sort_values("_y_raw").index.tolist()
                adjusted = top.loc[idx, "_y_raw"].to_numpy(dtype=float)
                for i in range(1, len(adjusted)):
                    if adjusted[i] - adjusted[i - 1] < min_gap:
                        adjusted[i] = adjusted[i - 1] + min_gap
                top.loc[idx, "_y"] = adjusted

    plot_figsize = figsize
    try:
        wm = float(width_multiplier)
        if wm > 0:
            plot_figsize = (figsize[0] * wm, figsize[1])
    except Exception:
        pass

    _, ax = plt.subplots(figsize=plot_figsize)
    ax.scatter(top["_x"], top["_y_raw"], s=14, color="#777777", alpha=0.35, zorder=1)

    # Keep labels flush-left per period (anchored at each year mark), while
    # connector vertical segments are parallel and slightly offset by rank.
    # `connector_dx_space` expands the marker->text gap used to distribute
    # connector x offsets, which helps avoid overcrowding.
    base_text_offset = float(label_x_base_offset) + max(0.0, float(connector_dx_space))
    top["_x_text"] = top["_x"].astype(float) + base_text_offset
    top["_x_marker"] = top["_x"].astype(float) + float(label_x_pad)
    top["_x_conn"] = top["_x_marker"]

    # Keep connector offsets fully inside marker->text gap to avoid overlap
    # with labels, while preserving parallel/offset vertical segments.
    for _, grp in top.groupby(period_col, sort=False):
        idx = grp.sort_values("_rank").index.tolist()
        n = len(idx)
        if n == 0:
            continue

        x_marker = float(top.loc[idx[0], "_x_marker"])
        x_text = float(top.loc[idx[0], "_x_text"])
        x_left = x_marker + float(label_x_pad)
        x_right = x_text - float(label_x_pad)

        if n == 1:
            x_conn_vals = [max(x_left, min(x_right, x_left))]
        else:
            available_span = max(0.0, x_right - x_left)
            desired_step = max(0.0, float(label_x_step))
            desired_span = desired_step * float(n - 1)
            if desired_span <= available_span:
                step = desired_step
            else:
                step = available_span / float(n - 1) if n > 1 else 0.0
            x_conn_vals = [x_left + i * step for i in range(n)]

        top.loc[idx, "_x_conn"] = x_conn_vals

    for _, row in top.sort_values(["_x", "_y"]).iterrows():
        x_marker = float(row["_x_marker"])
        x_conn = float(row["_x_conn"])
        x_label_edge = float(row["_x_text"]) - float(label_x_pad)

        if x_conn > x_marker:
            ax.plot(
                [x_marker, x_conn],
                [row["_y_raw"], row["_y_raw"]],
                color="#9a9a9a",
                lw=0.8,
                alpha=0.9,
                zorder=1,
            )
        ax.plot(
            [x_conn, x_conn],
            [row["_y_raw"], row["_y"]],
            color="#9a9a9a",
            lw=0.8,
            alpha=0.9,
            zorder=1,
        )
        if x_label_edge > x_conn:
            ax.plot(
                [x_conn, x_label_edge],
                [row["_y"], row["_y"]],
                color="#9a9a9a",
                lw=0.8,
                alpha=0.9,
                zorder=1,
            )
        ax.text(
            row["_x_text"],
            row["_y"],
            row["pair_label"],
            va="center",
            ha="left",
            fontsize=fontsize,
            zorder=2,
        )

    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha="right")
    extra_right = base_text_offset + 0.8
    ax.set_xlim(-0.4, max(len(periods) - 1 + extra_right, 0.6))

    if use_rank_fallback:
        ax.invert_yaxis()

    ax.set_xlabel("Period")
    ax.set_ylabel(y_label)
    ax.set_title(f"Top {top_k} (antibiotic, pathogen) pairs per period by {value_col}")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()
    plt.show()

    out = top.copy()
    out["period"] = out[period_col].astype(str)
    out["rank"] = pd.to_numeric(out["_rank"], errors="coerce").astype("Int64")
    out["value_raw"] = pd.to_numeric(out[value_col], errors="coerce")
    out["y_raw"] = pd.to_numeric(out["_y_raw"], errors="coerce")
    out["y_plot"] = pd.to_numeric(out["_y"], errors="coerce")
    out["x_idx"] = pd.to_numeric(out["_x"], errors="coerce")
    out["x_marker"] = pd.to_numeric(out["_x_marker"], errors="coerce")
    out["x_conn"] = pd.to_numeric(out["_x_conn"], errors="coerce")
    out["x_text"] = pd.to_numeric(out["_x_text"], errors="coerce")
    out = out[result_cols].sort_values(["x_idx", "rank", "antibiotic", "pathogen"]).reset_index(drop=True)
    return out


def compute_pair_trend_summary(
    norm_df: pd.DataFrame,
    amr_df: pd.DataFrame,
    pa_counts_df: pd.DataFrame,
    ab_counts_df: pd.DataFrame,
    resid_col: str = "resid_log_bg",
    z_col: str = "z_bg",
    amr_count_col: str = "count",
    min_points_for_slope: int = 2,
    log_x_offset: float = 1.0,
    start_year: int | None = None,
    end_year: int | None = None,
):
    """
    Build pair-level trend summary dataframe.

    Optional year window filtering is inclusive:
    - start_year=2000 means rows intersecting 2000-01-01 and later are kept.
    - end_year=2025 means rows intersecting up to 2025-12-31 are kept.

    Output columns:
    - antibiotic
    - pathogen
    - coef_resid_log_bg
    - coef_z_bg
    - amr_pair_total_count
    - pathogen_total_publications
    - antibiotic_total_publications
    - x_log_pair_total_count
    - y_coef_resid_log_bg
    - pair_label
    """
    required_norm = {"pathogen", "antibiotic", resid_col, z_col}
    missing_norm = required_norm - set(norm_df.columns)
    if missing_norm:
        raise ValueError(f"norm_df missing required columns: {sorted(missing_norm)}")

    required_amr = {"pathogen", "antibiotic", amr_count_col}
    missing_amr = required_amr - set(amr_df.columns)
    if missing_amr:
        raise ValueError(f"amr_df missing required columns: {sorted(missing_amr)}")

    required_pa = {"pathogen", "count"}
    missing_pa = required_pa - set(pa_counts_df.columns)
    if missing_pa:
        raise ValueError(f"pa_counts_df missing required columns: {sorted(missing_pa)}")

    required_ab = {"antibiotic", "count"}
    missing_ab = required_ab - set(ab_counts_df.columns)
    if missing_ab:
        raise ValueError(f"ab_counts_df missing required columns: {sorted(missing_ab)}")

    if start_year is not None and end_year is not None and int(start_year) > int(end_year):
        raise ValueError(f"start_year ({start_year}) must be <= end_year ({end_year}).")

    def _filter_year_window(
        data: pd.DataFrame,
        frame_name: str,
        require_date_cols: bool = True,
    ) -> pd.DataFrame:
        if start_year is None and end_year is None:
            return data.copy()
        if not {"start_date", "end_date"}.issubset(data.columns):
            if not require_date_cols:
                return data.copy()
            raise ValueError(
                f"{frame_name} must include 'start_date' and 'end_date' when "
                "start_year/end_year filtering is requested."
            )

        out = data.copy()
        out["_start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
        out["_end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
        out = out.dropna(subset=["_start_date", "_end_date"])

        # If one bound is omitted, expand to dataset extrema.
        effective_start = (
            pd.Timestamp(year=int(start_year), month=1, day=1)
            if start_year is not None
            else out["_start_date"].min()
        )
        effective_end = (
            pd.Timestamp(year=int(end_year), month=12, day=31)
            if end_year is not None
            else out["_end_date"].max()
        )
        out = out[(out["_end_date"] >= effective_start) & (out["_start_date"] <= effective_end)]

        return out.drop(columns=["_start_date", "_end_date"])

    df = _filter_year_window(norm_df, "norm_df")
    df, interval_col, _ = _resolve_interval_column(df)

    # Build a numeric time axis for regression (ordered intervals).
    if {"start_date", "end_date"}.issubset(df.columns):
        order_df = df[[interval_col, "start_date", "end_date"]].drop_duplicates().copy()
        order_df["_s"] = pd.to_datetime(order_df["start_date"], errors="coerce")
        order_df["_e"] = pd.to_datetime(order_df["end_date"], errors="coerce")
        order_df = order_df.sort_values(["_s", "_e", interval_col])
        interval_order = order_df[interval_col].astype(str).tolist()
    else:
        interval_order = sorted(df[interval_col].astype(str).unique().tolist())
    interval_to_x = {k: i for i, k in enumerate(interval_order)}

    # Ensure numeric regression targets and clean rows.
    df[resid_col] = pd.to_numeric(df[resid_col], errors="coerce")
    df[z_col] = pd.to_numeric(df[z_col], errors="coerce")
    df = df.dropna(subset=["pathogen", "antibiotic"])
    df["_interval_key"] = df[interval_col].astype(str)
    df["_x_time"] = df["_interval_key"].map(interval_to_x).astype(float)

    pair_keys = ["antibiotic", "pathogen"]

    def _slope_for_group(group: pd.DataFrame, value_col: str) -> float:
        gx = pd.to_numeric(group["_x_time"], errors="coerce")
        gy = pd.to_numeric(group[value_col], errors="coerce")
        ok = np.isfinite(gx) & np.isfinite(gy)
        if int(ok.sum()) < int(min_points_for_slope):
            return float("nan")
        x = gx[ok].to_numpy(dtype=float)
        y = gy[ok].to_numpy(dtype=float)
        if np.unique(x).size < 2:
            return float("nan")
        return float(np.polyfit(x, y, 1)[0])

    grouped = df.groupby(pair_keys, as_index=False, sort=False)
    coef_rows = []
    for (ab, pa), g in grouped:
        coef_rows.append(
            {
                "antibiotic": str(ab),
                "pathogen": str(pa),
                "coef_resid_log_bg": _slope_for_group(g, resid_col),
                "coef_z_bg": _slope_for_group(g, z_col),
            }
        )
    coef_df = pd.DataFrame(coef_rows)

    amr_filtered = _filter_year_window(amr_df, "amr_df")
    amr_totals = (
        amr_filtered
        .assign(
            pathogen=lambda d: d["pathogen"].astype(str),
            antibiotic=lambda d: d["antibiotic"].astype(str),
            _count_num=lambda d: pd.to_numeric(d[amr_count_col], errors="coerce"),
        )
        .dropna(subset=["_count_num"])
        .groupby(pair_keys, as_index=False)["_count_num"]
        .sum()
        .rename(columns={"_count_num": "amr_pair_total_count"})
    )

    pa_totals = (
        _filter_year_window(pa_counts_df, "pa_counts_df", require_date_cols=False)
        .assign(
            pathogen=lambda d: d["pathogen"].astype(str),
            _count_num=lambda d: pd.to_numeric(d["count"], errors="coerce"),
        )
        .dropna(subset=["_count_num"])
        .groupby("pathogen", as_index=False)["_count_num"]
        .sum()
        .rename(columns={"_count_num": "pathogen_total_publications"})
    )

    ab_totals = (
        _filter_year_window(ab_counts_df, "ab_counts_df", require_date_cols=False)
        .assign(
            antibiotic=lambda d: d["antibiotic"].astype(str),
            _count_num=lambda d: pd.to_numeric(d["count"], errors="coerce"),
        )
        .dropna(subset=["_count_num"])
        .groupby("antibiotic", as_index=False)["_count_num"]
        .sum()
        .rename(columns={"_count_num": "antibiotic_total_publications"})
    )

    summary_df = (
        coef_df.merge(amr_totals, on=pair_keys, how="left")
        .merge(pa_totals, on="pathogen", how="left")
        .merge(ab_totals, on="antibiotic", how="left")
    )

    summary_df["amr_pair_total_count"] = pd.to_numeric(
        summary_df["amr_pair_total_count"], errors="coerce"
    ).fillna(0.0)
    summary_df["x_log_pair_total_count"] = np.log(summary_df["amr_pair_total_count"] + float(log_x_offset))
    summary_df["y_coef_resid_log_bg"] = pd.to_numeric(
        summary_df["coef_resid_log_bg"], errors="coerce"
    )
    summary_df["pair_label"] = (
        summary_df["antibiotic"].astype(str) + "\n" + summary_df["pathogen"].astype(str)
    )

    return summary_df


def _pareto_front_shells(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_shells: int | None = None,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> pd.Series:
    """
    Compute Pareto shell index (1=best) with configurable objectives.
    Returns a series aligned with df.index; NaN for rows with non-finite x/y.
    """
    shells = pd.Series(np.nan, index=df.index, dtype=float)
    valid = df[np.isfinite(pd.to_numeric(df[x_col], errors="coerce")) & np.isfinite(pd.to_numeric(df[y_col], errors="coerce"))]
    remaining = valid.index.tolist()
    shell_id = 1

    while remaining and (n_shells is None or shell_id <= int(n_shells)):
        rem = df.loc[remaining, [x_col, y_col]].astype(float)
        pts = rem.to_numpy()
        tx = pts[:, 0] if maximize_x else -pts[:, 0]
        ty = pts[:, 1] if maximize_y else -pts[:, 1]
        nondominated = np.ones(len(rem), dtype=bool)

        # i is dominated if any j is better-or-equal on both transformed axes
        # (maximize) and strictly better on at least one axis.
        for i in range(len(rem)):
            dominates_i = (
                (tx >= tx[i])
                & (ty >= ty[i])
                & ((tx > tx[i]) | (ty > ty[i]))
            )
            if np.any(dominates_i):
                nondominated[i] = False

        shell_indices = rem.index[nondominated].tolist()
        if not shell_indices:
            break
        shells.loc[shell_indices] = float(shell_id)
        shell_set = set(shell_indices)
        remaining = [idx for idx in remaining if idx not in shell_set]
        shell_id += 1

    return shells


def plot_pair_trend_summary(
    pair_trend_df: pd.DataFrame,
    n_pareto_front_shells: int | tuple[int, int] | None = None,
    x_col: str = "x_log_pair_total_count",
    y_col: str = "y_coef_resid_log_bg",
    label_col: str = "pair_label",
    figsize: tuple[int, int] = (14, 10),
    annotate: bool = True,
    fontsize: int = 7,
    show_pareto_front_lines: bool = True,
    front_linewidth: float = 2.0,
    pareto_cmap: str = "Pastel1",
    use_lowest_coef_front: bool = False,
    min_n_pubs: float = 100,
    min_reg_coef: float = 1,
):
    """
    Plot pair trend summary in 2D.

    If n_pareto_front_shells is:
    - int N: only points in shells 1..N are shown (existing behavior).
    - tuple (a, b): only points in shells a..b are shown.
    Optional colored front lines can be overlaid via show_pareto_front_lines.

    Set use_lowest_coef_front=True to select shells by:
    - maximize x_col (log pair count)
    - minimize y_col (coefficient)

    Background mask:
    - Gray where constraints are not met.
    - White where both are met: n_pubs >= min_n_pubs and coef >= min_reg_coef.
    """
    required_cols = {x_col, y_col, label_col}
    missing = required_cols - set(pair_trend_df.columns)
    if missing:
        raise ValueError(f"pair_trend_df missing required columns: {sorted(missing)}")

    # Fallback for stale interactive sessions where global helper resolution can fail.
    def _pareto_front_shells_local(
        df_local: pd.DataFrame,
        x_name: str,
        y_name: str,
        n_shells_local: int | None = None,
        maximize_x_local: bool = True,
        maximize_y_local: bool = True,
    ) -> pd.Series:
        shells_local = pd.Series(np.nan, index=df_local.index, dtype=float)
        valid_local = df_local[
            np.isfinite(pd.to_numeric(df_local[x_name], errors="coerce"))
            & np.isfinite(pd.to_numeric(df_local[y_name], errors="coerce"))
        ]
        remaining_local = valid_local.index.tolist()
        shell_id_local = 1

        while remaining_local and (n_shells_local is None or shell_id_local <= int(n_shells_local)):
            rem_local = df_local.loc[remaining_local, [x_name, y_name]].astype(float)
            pts_local = rem_local.to_numpy()
            tx_local = pts_local[:, 0] if maximize_x_local else -pts_local[:, 0]
            ty_local = pts_local[:, 1] if maximize_y_local else -pts_local[:, 1]
            nondominated_local = np.ones(len(rem_local), dtype=bool)

            for i in range(len(rem_local)):
                dominates_i_local = (
                    (tx_local >= tx_local[i])
                    & (ty_local >= ty_local[i])
                    & ((tx_local > tx_local[i]) | (ty_local > ty_local[i]))
                )
                if np.any(dominates_i_local):
                    nondominated_local[i] = False

            shell_indices_local = rem_local.index[nondominated_local].tolist()
            if not shell_indices_local:
                break
            shells_local.loc[shell_indices_local] = float(shell_id_local)
            shell_set_local = set(shell_indices_local)
            remaining_local = [idx for idx in remaining_local if idx not in shell_set_local]
            shell_id_local += 1

        return shells_local

    plot_df = pair_trend_df.copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])

    if plot_df.empty:
        print("No finite points to plot.")
        return plot_df

    shell_start: int | None = None
    shell_end: int | None = None
    if isinstance(n_pareto_front_shells, tuple):
        if len(n_pareto_front_shells) != 2:
            raise ValueError("n_pareto_front_shells tuple must be (start, end).")
        shell_start = int(n_pareto_front_shells[0])
        shell_end = int(n_pareto_front_shells[1])
        if shell_start < 1 or shell_end < shell_start:
            raise ValueError("For tuple n_pareto_front_shells=(start,end), require 1 <= start <= end.")
    elif n_pareto_front_shells is not None:
        shell_end = int(n_pareto_front_shells)
        shell_start = 1
        if shell_end < 1:
            raise ValueError("n_pareto_front_shells must be >= 1 when provided.")

    maximize_y = not bool(use_lowest_coef_front)
    pareto_fn = globals().get("_pareto_front_shells", _pareto_front_shells_local)
    plot_df["pareto_shell"] = pareto_fn(
        plot_df,
        x_col,
        y_col,
        shell_end,
        True,
        maximize_y,
    )

    if shell_end is not None:
        plot_df = plot_df[
            plot_df["pareto_shell"].notna()
            & (plot_df["pareto_shell"] >= int(shell_start))
            & (plot_df["pareto_shell"] <= int(shell_end))
        ].copy()

    _, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # Background gating region: assume x_col is a log pair-count axis.
    # Valid area is x >= log(min_n_pubs) and y >= min_reg_coef.
    x_vals = plot_df[x_col].to_numpy(dtype=float)
    y_vals = plot_df[y_col].to_numpy(dtype=float)
    x_pad = max(1e-6, 0.04 * (float(np.nanmax(x_vals)) - float(np.nanmin(x_vals)) or 1.0))
    y_pad = max(1e-6, 0.06 * (float(np.nanmax(y_vals)) - float(np.nanmin(y_vals)) or 1.0))
    x_min = float(np.nanmin(x_vals)) - x_pad
    x_max = float(np.nanmax(x_vals)) + x_pad
    y_min = float(np.nanmin(y_vals)) - y_pad
    y_max = float(np.nanmax(y_vals)) + y_pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    bg_gray = "#ececec"
    if min_n_pubs is not None:
        x_threshold = float(np.log(max(float(min_n_pubs), 1e-12)))
        x_left = min(max(x_threshold, x_min), x_max)
        if x_left > x_min:
            ax.axvspan(x_min, x_left, color=bg_gray, alpha=1.0, zorder=0)
    else:
        x_threshold = x_min
    if min_reg_coef is not None and x_max > x_min:
        y_threshold = float(min_reg_coef)
        y_low = min(max(y_threshold, y_min), y_max)
        if y_low > y_min:
            frac_start = float(np.clip((x_threshold - x_min) / (x_max - x_min), 0.0, 1.0))
            ax.axhspan(y_min, y_low, xmin=frac_start, xmax=1.0, color=bg_gray, alpha=1.0, zorder=0)

    if shell_end is None:
        ax.scatter(plot_df[x_col], plot_df[y_col], s=14, color="#bdbdbd", alpha=0.5, zorder=1)
    else:
        shell_vals = plot_df["pareto_shell"].astype(int).to_numpy()
        sc = ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            c=shell_vals,
            cmap=pareto_cmap,
            s=18,
            alpha=0.85,
            zorder=1,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Pareto shell (1=best)")

    if show_pareto_front_lines:
        shell_unique = sorted(plot_df["pareto_shell"].dropna().astype(int).unique().tolist())
        if shell_unique:
            cmap = plt.get_cmap(pareto_cmap)
            max_shell = max(shell_unique)
            def _plot_step_front(ax_obj, x_vals, y_vals, color_val, lw_val):
                # Draw Pareto front as staircase: vertical first, then horizontal.
                if len(x_vals) < 2:
                    return
                for i in range(len(x_vals) - 1):
                    x0 = float(x_vals[i])
                    y0 = float(y_vals[i])
                    x1 = float(x_vals[i + 1])
                    y1 = float(y_vals[i + 1])
                    # Vertical segment at current x
                    ax_obj.plot(
                        [x0, x0],
                        [y0, y1],
                        color=color_val,
                        linewidth=lw_val,
                        alpha=0.9,
                        zorder=3,
                    )
                    # Horizontal segment at next y
                    ax_obj.plot(
                        [x0, x1],
                        [y1, y1],
                        color=color_val,
                        linewidth=lw_val,
                        alpha=0.9,
                        zorder=3,
                    )

            for shell in shell_unique:
                shell_df = plot_df[plot_df["pareto_shell"].astype(int) == shell].copy()
                if len(shell_df) < 2:
                    continue
                shell_df = (
                    shell_df[[x_col, y_col]]
                    .dropna()
                    .groupby(x_col, as_index=False)[y_col]
                    .agg("min" if use_lowest_coef_front else "max")
                    .sort_values(x_col)
                )
                if len(shell_df) < 2:
                    continue
                if shell_end is None:
                    color = "#fbb4ae" if shell == 1 else "#ccebc5"
                else:
                    denom = max(1, max_shell - 1)
                    color = cmap((shell - 1) / denom)
                xs = shell_df[x_col].to_numpy(dtype=float)
                ys = shell_df[y_col].to_numpy(dtype=float)
                _plot_step_front(
                    ax_obj=ax,
                    x_vals=xs,
                    y_vals=ys,
                    color_val=color,
                    lw_val=float(front_linewidth),
                )

    if annotate:
        for _, row in plot_df.iterrows():
            ax.text(
                float(row[x_col]),
                float(row[y_col]),
                str(row[label_col]),
                fontsize=fontsize,
                ha="left",
                va="center",
                zorder=5,
            )

    ax.set_xlabel("log(total AMR pair count)")
    ax.set_ylabel("Linear regression coefficient (resid_log_bg)")
    if shell_end is None:
        mode_txt = "lowest coef" if use_lowest_coef_front else "highest coef"
        ax.set_title(f"Pair trend map: log pair count vs resid_log_bg slope ({mode_txt} front)")
    else:
        mode_txt = "lowest coef" if use_lowest_coef_front else "highest coef"
        if shell_start == 1:
            shell_txt = f"1..{int(shell_end)}"
        else:
            shell_txt = f"{int(shell_start)}..{int(shell_end)}"
        ax.set_title(
            f"Pair trend map (Pareto shells {shell_txt}, {mode_txt} front)"
        )
    ax.grid(alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.show()

    sort_cols = []
    ascending = []
    if "pareto_shell" in plot_df.columns:
        sort_cols.append("pareto_shell")
        ascending.append(True)
    if "amr_pair_total_count" in plot_df.columns:
        sort_cols.append("amr_pair_total_count")
        ascending.append(True)
    if sort_cols:
        plot_df = plot_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return plot_df


def build_pair_trend_summary(
    norm_df: pd.DataFrame,
    amr_df: pd.DataFrame,
    pa_counts_df: pd.DataFrame,
    ab_counts_df: pd.DataFrame,
    resid_col: str = "resid_log_bg",
    z_col: str = "z_bg",
    amr_count_col: str = "count",
    min_points_for_slope: int = 2,
    log_x_offset: float = 1.0,
    start_year: int | None = None,
    end_year: int | None = None,
    figsize: tuple[int, int] = (14, 10),
    annotate: bool = True,
    fontsize: int = 7,
):
    """
    Backward-compatible wrapper.
    Prefer compute_pair_trend_summary + plot_pair_trend_summary.
    """
    summary_df = compute_pair_trend_summary(
        norm_df=norm_df,
        amr_df=amr_df,
        pa_counts_df=pa_counts_df,
        ab_counts_df=ab_counts_df,
        resid_col=resid_col,
        z_col=z_col,
        amr_count_col=amr_count_col,
        min_points_for_slope=min_points_for_slope,
        log_x_offset=log_x_offset,
        start_year=start_year,
        end_year=end_year,
    )
    plot_pair_trend_summary(
        summary_df,
        n_pareto_front_shells=None,
        x_col="x_log_pair_total_count",
        y_col="y_coef_resid_log_bg",
        label_col="pair_label",
        figsize=figsize,
        annotate=annotate,
        fontsize=fontsize,
    )
    return summary_df


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
    ax = sns.heatmap(
        pivot,
        cmap="RdBu_r",
        norm=norm,
        cbar_kws={"label": f"{value} (detrended)"},
        linewidths=0.3,
    )
    # Show interval labels on both bottom and top for easier reading
    try:
        ax.tick_params(axis="x", top=True, bottom=True, labeltop=True, labelbottom=True)
    except Exception:
        pass
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
    color_envelope: tuple[float, float] | None = None,
    mp4_path: str | None = None,
    video_total_duration: float | None = None,
    video_ping_pong: bool = False,
    video_loop: bool = True,
    image_size: tuple[int, int] | None = (704, 512),
):
    """Project pathogen embeddings with UMAP and color by antibiotic activity.

    Parameters
    - color_envelope: optional (real, visualized) pivot on [0,1].
      If provided, apply a piecewise linear remapping of normalized color
      values x in [0,1] so that x=real maps to visualized, enhancing lower
      values: for x <= real: y = (visualized/real) * x; for x > real:
      y = visualized + (1-visualized) * (x - real) / (1 - real).
      The remapped values are then scaled back to the original [0, overall_max]
      range so the colorbar remains in data units.
    """
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

    def _apply_piecewise_envelope(values: pd.Series, max_val: float) -> pd.Series:
        if color_envelope is None or values.empty:
            return values
        r, v = color_envelope
        # Validate and clamp parameters to avoid division errors
        r = float(np.clip(r, 1e-9, 1 - 1e-9))
        v = float(np.clip(v, 0.0, 1.0))
        # Normalize to [0,1]
        norm = (values / max_val) if max_val > 0 else values.copy()
        # Piecewise mapping
        below = norm <= r
        out = norm.copy()
        if r > 0:
            out[below] = (v / r) * norm[below]
        else:
            out[below] = 0.0
        if r < 1:
            out[~below] = v + (1.0 - v) * (norm[~below] - r) / (1.0 - r)
        else:
            out[~below] = 1.0
        # Rescale back to data units so colorbar remains meaningful
        return out * max_val

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
            display_series = _apply_piecewise_envelope(color_series, overall_max)
            frac = 0.0 if base_frame_count <= 1 else idx / (base_frame_count - 1)
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=working["umap_x"],
                        y=working["umap_y"],
                        mode="markers",
                        marker=dict(
                            size=marker_pixel_sizes,
                            color=display_series,
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
        display_series = _apply_piecewise_envelope(color_series, overall_max)
        fig.add_trace(
            go.Scatter(
                x=working["umap_x"],
                y=working["umap_y"],
                mode="markers",
                marker=dict(
                    size=marker_pixel_sizes,
                    color=display_series,
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
        autosize=False if image_size else True,
        width=(image_size[0] if image_size else 500 * cols),
        height=(image_size[1] if image_size else 500 * rows),
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


def build_pubmed_links_html(
    amr_df: pd.DataFrame,
    pathogen: str,
    antibiotic: str,
    date,
) -> str:
    """
    Return an HTML string listing PubMed links for PMIDs matching
    the given pathogen, antibiotic, and a target date within [start_date, end_date].

    The function supports two input shapes:
    - Interval rows with a list-like "pmids" column.
    - Per-PMID rows with a scalar "pmid" column.

    Usage in a notebook:
        from IPython.display import HTML
        html = build_pubmed_links_html(amr_df, "Neisseria gonorrhoeae", "Thiamphenicol", "1980/01/01")
        HTML(html)
    """
    if not {"pathogen", "antibiotic", "start_date", "end_date"}.issubset(amr_df.columns):
        raise ValueError("amr_df must include columns: pathogen, antibiotic, start_date, end_date")

    target_date = pd.to_datetime(date)
    df = amr_df.copy()
    df["_path"] = df["pathogen"].astype(str).str.strip().str.lower()
    df["_abx"] = df["antibiotic"].astype(str).str.strip().str.lower()
    want_path = str(pathogen).strip().lower()
    want_abx = str(antibiotic).strip().lower()
    df["start_date"] = pd.to_datetime(df["start_date"])  # errors raise
    df["end_date"] = pd.to_datetime(df["end_date"])      # errors raise

    mask = (df["_path"] == want_path) & (df["_abx"] == want_abx) & (df["start_date"] <= target_date) & (df["end_date"] >= target_date)
    sub = df.loc[mask].copy()

    pmid_list: list[str] = []
    if "pmids" in sub.columns:
        def _as_list(x):
            if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
                return [str(i) for i in x]
            if pd.isna(x) or x is None:
                return []
            # try to parse common string representations like "[123, 456]"
            try:
                import ast as _ast
                val = _ast.literal_eval(str(x))
                if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
                    return [str(i) for i in val]
            except Exception:
                pass
            return [str(x)]

        for items in sub["pmids"].tolist():
            pmid_list.extend(_as_list(items))
    elif "pmid" in sub.columns:
        pmid_list = [str(v) for v in sub["pmid"].dropna().astype(str).tolist()]
    else:
        # No PMIDs column available
        return "<em>No PMIDs column found (expected 'pmids' or 'pmid').</em>"

    # Normalize and unique while preserving order
    seen = set()
    unique_pmids: list[str] = []
    for p in pmid_list:
        p_stripped = "".join(ch for ch in str(p) if ch.isdigit())
        if not p_stripped or p_stripped in seen:
            continue
        seen.add(p_stripped)
        unique_pmids.append(p_stripped)

    header = f"<strong>{pathogen}</strong> × <strong>{antibiotic}</strong> on {pd.to_datetime(target_date).date()}: {len(unique_pmids)} PMIDs"
    if not unique_pmids:
        return header + "<br><em>No PMIDs found for the selected date.</em>"

    items = "\n".join(
        f'<li><a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank" rel="noopener noreferrer">{pmid}</a></li>'
        for pmid in unique_pmids
    )
    return f"{header}<ul>{items}</ul>"


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
    "plot_venue_vector",
    "plot_top_antibiotic_period_by_pathogen",
    "build_pubmed_links_html",
    "_umap_scatter",
]
