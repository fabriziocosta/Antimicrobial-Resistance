"""Debiasing utilities for AMR counts and pathogen embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _infer_interval_span(start_year: pd.Series, end_year: pd.Series):
    """Infer the most common inclusive interval span from the data."""
    spans = (end_year - start_year).dropna() + 1
    spans = spans[spans > 0]
    if spans.empty:
        return None
    mode = spans.mode()
    if mode.empty:
        return None
    return int(mode.iloc[0])


def _label_interval(start_year, end_year, span_years=None, method: str = "end"):
    """Assign an interval label using the specified method and span."""
    if pd.isna(start_year) or pd.isna(end_year):
        return None

    start_year = int(start_year)
    end_year = int(end_year)

    if span_years is None or span_years <= 0:
        label_start = start_year
        label_end = end_year
    else:
        span_years = int(span_years)
        if span_years <= 0:
            raise ValueError("span_years must be positive when provided.")

        if method == "end":
            anchor = end_year
            label_start = (anchor // span_years) * span_years
        elif method == "start":
            anchor = start_year
            label_start = (anchor // span_years) * span_years
        elif method == "midpoint":
            mid = (start_year + end_year) / 2.0
            label_start = (int(mid) // span_years) * span_years
        elif method == "max_overlap":
            d0 = (start_year // span_years) * span_years
            d1 = (end_year // span_years) * span_years
            best_start, best_len = d0, -1
            for d in range(d0, d1 + span_years, span_years):
                seg_start = max(start_year, d)
                seg_end = min(end_year, d + span_years - 1)
                overlap = max(0, seg_end - seg_start + 1)
                if overlap > best_len:
                    best_len, best_start = overlap, d
            label_start = best_start
        else:
            raise ValueError("method must be one of {'end','start','midpoint','max_overlap'}")

        label_end = label_start + span_years - 1
        start_year, end_year = label_start, label_end

    return f"{start_year}-{end_year}"


def _compute_marginal_distribution(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Return marginal log-count distribution per interval for a given key."""
    group_cols = [key_col, "interval"]
    marginal = df.groupby(group_cols, as_index=False)["count"].sum().rename(columns={"count": "count_sum"})
    marginal["marginal_log"] = np.log1p(marginal["count_sum"])
    marginal["total_marginal_log"] = marginal.groupby(key_col)["marginal_log"].transform("sum")
    marginal["marginal_prob"] = np.where(
        marginal["total_marginal_log"] > 0,
        marginal["marginal_log"] / marginal["total_marginal_log"],
        0.0,
    )
    return marginal[[key_col, "interval", "marginal_prob", "marginal_log", "total_marginal_log"]]


def _prepare_pairwise_interval_counts(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    group_cols = ["pathogen", key_col, "interval"]
    pairwise = df.groupby(group_cols, as_index=False)["count"].sum()
    pairwise["log_count"] = np.log1p(pairwise["count"])
    pairwise["total_count"] = pairwise.groupby(["pathogen", key_col])["count"].transform("sum")
    pairwise["total_log_count"] = np.log1p(pairwise["total_count"])
    return pairwise


def debias_amr_marginal(
    amr_df: pd.DataFrame,
    class_df: pd.DataFrame,
    interval_method: str = "end",
    interval_years: int | None = None,
):
    """Compute marginal log-scale debiasing for antibiotic and class levels."""
    df = amr_df.copy()
    df["antibiotic"] = df["antibiotic"].str.lower()

    start_year = pd.to_datetime(df["start_date"]).dt.year
    end_year = pd.to_datetime(df["end_date"]).dt.year
    interval_span = interval_years or _infer_interval_span(start_year, end_year)
    if interval_span is None:
        interval_span = 10

    df["interval"] = [
        _label_interval(s, e, span_years=interval_span, method=interval_method) for s, e in zip(start_year, end_year)
    ]
    df["log_count"] = np.log1p(df["count"])

    ab_marginal = _compute_marginal_distribution(df, "antibiotic")
    ab_counts = _prepare_pairwise_interval_counts(df, "antibiotic")
    ab_totals = ab_counts[["pathogen", "antibiotic", "total_count", "total_log_count"]].drop_duplicates()

    ab_full = ab_totals.merge(ab_marginal[["antibiotic", "interval", "marginal_prob"]], on="antibiotic", how="left")
    ab_full = ab_full.merge(
        ab_counts[["pathogen", "antibiotic", "interval", "count", "log_count"]],
        on=["pathogen", "antibiotic", "interval"],
        how="left",
    )

    ab_full["count"] = ab_full["count"].fillna(0.0)
    ab_full["log_count"] = ab_full["log_count"].fillna(np.log1p(ab_full["count"]))
    ab_full["marginal_prob"] = ab_full["marginal_prob"].fillna(0.0)

    ab_full["expected_count"] = ab_full["total_count"] * ab_full["marginal_prob"]
    ab_full["expected_log"] = np.log1p(ab_full["expected_count"])
    ab_full["diff"] = ab_full["log_count"] - ab_full["expected_log"]

    ab_full = ab_full[
        [
            "pathogen",
            "antibiotic",
            "interval",
            "count",
            "log_count",
            "total_count",
            "total_log_count",
            "marginal_prob",
            "expected_log",
            "expected_count",
            "diff",
        ]
    ]

    class_df = class_df.rename(columns={c: c.lower() for c in class_df.columns}).copy()
    class_df["antibiotic"] = class_df["antibiotic"].str.lower()
    df_class = df.merge(class_df, on="antibiotic", how="left")
    df_class["class"] = df_class["class"].fillna(df_class["antibiotic"])

    cl_marginal = _compute_marginal_distribution(df_class, "class")
    cl_counts = _prepare_pairwise_interval_counts(df_class, "class")
    cl_totals = cl_counts[["pathogen", "class", "total_count", "total_log_count"]].drop_duplicates()

    cl_full = cl_totals.merge(cl_marginal[["class", "interval", "marginal_prob"]], on="class", how="left")
    cl_full = cl_full.merge(
        cl_counts[["pathogen", "class", "interval", "count", "log_count"]],
        on=["pathogen", "class", "interval"],
        how="left",
    )

    cl_full["count"] = cl_full["count"].fillna(0.0)
    cl_full["log_count"] = cl_full["log_count"].fillna(np.log1p(cl_full["count"]))
    cl_full["marginal_prob"] = cl_full["marginal_prob"].fillna(0.0)

    cl_full["expected_count"] = cl_full["total_count"] * cl_full["marginal_prob"]
    cl_full["expected_log"] = np.log1p(cl_full["expected_count"])
    cl_full["diff"] = cl_full["log_count"] - cl_full["expected_log"]

    cl_full = cl_full[
        [
            "pathogen",
            "class",
            "interval",
            "count",
            "log_count",
            "total_count",
            "total_log_count",
            "marginal_prob",
            "expected_log",
            "expected_count",
            "diff",
        ]
    ]

    return ab_full, cl_full


def _build_pathogen_vectors(marg_df: pd.DataFrame, key_col: str):
    """Convert marginal debiasing results into pathogen vectors."""
    if marg_df.empty:
        pathogens = []
        keys = []
        intervals = []
    else:
        pathogens = sorted(marg_df["pathogen"].dropna().unique())
        keys = sorted(marg_df[key_col].dropna().unique())
        intervals = sorted(marg_df["interval"].dropna().unique())

    if keys and intervals:
        combination_index = pd.MultiIndex.from_product([keys, intervals], names=[key_col, "interval"])
    else:
        combination_index = pd.MultiIndex.from_arrays([[], []], names=[key_col, "interval"])

    vectors = []
    totals = []
    for pathogen in pathogens:
        sub = (
            marg_df[marg_df["pathogen"] == pathogen]
            .set_index([key_col, "interval"])[["log_count", "expected_log"]]
            .reindex(combination_index, fill_value=0.0)
        )
        total_log = sub["log_count"].sum()
        totals.append(total_log)
        norm_log = sub["log_count"] / total_log if total_log > 0 else sub["log_count"]
        expected_total = sub["expected_log"].sum()
        norm_expected = sub["expected_log"] / expected_total if expected_total > 0 else sub["expected_log"]
        residual = norm_log - norm_expected
        vectors.append(residual.to_numpy())

    matrix = np.array(vectors) if vectors else np.empty((0, len(combination_index)))
    return np.array(pathogens), np.array(totals), matrix


def build_pathogen_vectors_antibiotic(amr_df: pd.DataFrame, class_df: pd.DataFrame, interval_method: str = "end", interval_years=None):
    """Return pathogen vectors using antibiotic-level marginal debiasing."""
    marg_ab, _ = debias_amr_marginal(amr_df, class_df, interval_method=interval_method, interval_years=interval_years)
    return _build_pathogen_vectors(marg_ab, "antibiotic")


def build_pathogen_vectors_class(amr_df: pd.DataFrame, class_df: pd.DataFrame, interval_method: str = "end", interval_years=None):
    """Return pathogen vectors using class-level marginal debiasing."""
    _, marg_cl = debias_amr_marginal(amr_df, class_df, interval_method=interval_method, interval_years=interval_years)
    return _build_pathogen_vectors(marg_cl, "class")


__all__ = [
    "debias_amr_marginal",
    "build_pathogen_vectors_antibiotic",
    "build_pathogen_vectors_class",
]
