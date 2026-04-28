"""Minimal yearly scientific-interest utilities for the refactored workflow."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from amr.io import atomic_write_parquet

try:  # pragma: no cover - exercised only when optional dependency is installed
    from skmisc.loess import loess
except ImportError:  # pragma: no cover - environment-dependent fallback
    loess = None

_AUTO_TAIL_DAMPING = object()
WEAK_SIGNAL_CATEGORY = "CO_OCCURRENCE_OR_WEAK_SIGNAL"
AMR_DRIVER_CATEGORY_ORDER = [
    "RESISTANCE_PREVALENCE_SURVEILLANCE",
    "RESISTANCE_MECHANISM",
    "TREATMENT_EFFECTIVENESS",
    "TREATMENT_REUSE_OR_SALVAGE",
    "DIAGNOSTIC_STEWARDSHIP",
    "GUIDELINE_POLICY_STEWARDSHIP",
    "ONE_HEALTH_ENVIRONMENTAL_RESERVOIR",
    WEAK_SIGNAL_CATEGORY,
]
AMR_DRIVER_CATEGORY_COLORS = {
    "RESISTANCE_PREVALENCE_SURVEILLANCE": "#4e79a7",
    "RESISTANCE_MECHANISM": "#f28e2b",
    "TREATMENT_EFFECTIVENESS": "#59a14f",
    "TREATMENT_REUSE_OR_SALVAGE": "#e15759",
    "DIAGNOSTIC_STEWARDSHIP": "#76b7b2",
    "GUIDELINE_POLICY_STEWARDSHIP": "#edc948",
    "ONE_HEALTH_ENVIRONMENTAL_RESERVOIR": "#b07aa1",
    WEAK_SIGNAL_CATEGORY: "#9c755f",
}
AMR_DRIVER_CATEGORY_COLUMNS = [
    "pathogen",
    "antibiotic",
    "year",
    "amr_primary_category",
    "count",
]
AMR_DRIVER_SUMMARY_COLUMNS = [
    "pathogen",
    "antibiotic",
    "driver_recent_total",
    "driver_previous_total",
    "driver_count_change",
    "driver_dominant_category",
    "driver_dominant_category_count",
    "driver_fastest_growing_category",
    "driver_fastest_growing_category_change",
    "driver_weak_signal_share",
    "driver_manual_review_share",
    "driver_central_count",
    "driver_partial_count",
    "driver_incidental_count",
]


def _log_transform(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.log10(1.0 + values)


def _inverse_log_transform(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.power(10.0, values) - 1.0


def fit_global_pooled_si_model(
    amr_background_df: pd.DataFrame,
    *,
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, Ridge, list[str]]:
    """Fit pooled baseline on log-counts and define SI as residual."""
    required = {
        "pathogen",
        "antibiotic",
        "log_count",
        "log_bg_pathogen_n",
        "log_bg_antibiotic_n",
    }
    missing = required - set(amr_background_df.columns)
    if missing:
        raise KeyError(f"amr_background_df missing required columns: {sorted(missing)}")

    work = amr_background_df.copy()
    pathogen_dummies = pd.get_dummies(work["pathogen"].astype(str), prefix="pathogen", dtype=float)
    antibiotic_dummies = pd.get_dummies(work["antibiotic"].astype(str), prefix="antibiotic", dtype=float)
    design_df = pd.concat(
        [
            work[["log_bg_pathogen_n", "log_bg_antibiotic_n"]].reset_index(drop=True),
            pathogen_dummies.reset_index(drop=True),
            antibiotic_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    response = work["log_count"].to_numpy(dtype=float)

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design_df, response)

    work["expected_log_count_global"] = model.predict(design_df)
    work["expected_count_global"] = np.clip(_inverse_log_transform(work["expected_log_count_global"]), 0.0, None)
    work["SI"] = work["log_count"] - work["expected_log_count_global"]
    return work, model, design_df.columns.tolist()


def build_pair_si_yearly_series(si_df: pd.DataFrame, pathogen: str, antibiotic: str) -> pd.DataFrame:
    """Aggregate SI to one row per pathogen-antibiotic-year."""
    required = {
        "pathogen",
        "antibiotic",
        "year",
        "count",
        "bg_pathogen_n",
        "bg_antibiotic_n",
        "expected_count_global",
        "expected_log_count_global",
        "SI",
    }
    missing = required - set(si_df.columns)
    if missing:
        raise KeyError(f"si_df missing required yearly SI columns: {sorted(missing)}")

    pair_df = (
        si_df.loc[
            (si_df["pathogen"] == pathogen) & (si_df["antibiotic"] == antibiotic),
            [
                "pathogen",
                "antibiotic",
                "year",
                "count",
                "bg_pathogen_n",
                "bg_antibiotic_n",
                "expected_count_global",
                "expected_log_count_global",
                "SI",
            ],
        ]
        .sort_values("year")
        .groupby(["pathogen", "antibiotic", "year"], as_index=False)
        .agg(
            {
                "count": "sum",
                "bg_pathogen_n": "max",
                "bg_antibiotic_n": "max",
                "expected_count_global": "sum",
                "expected_log_count_global": "mean",
                "SI": "mean",
            }
        )
        .reset_index(drop=True)
    )
    if pair_df.empty:
        raise ValueError(f"No yearly SI data found for pathogen={pathogen!r}, antibiotic={antibiotic!r}.")
    pair_df["log_count"] = _log_transform(pair_df["count"].astype(float))
    return pair_df


def build_yearly_category_counts(
    unified_df: pd.DataFrame,
    *,
    include_decisions: tuple[str, ...] | list[str] | set[str] = ("yes", "uncertain"),
    category_col: str = "amr_primary_category",
) -> pd.DataFrame:
    """Aggregate unified LLM decisions to yearly AMR category counts."""
    required = {"pathogen", "antibiotic", "year", "unified_decision", category_col}
    if unified_df.empty or not required.issubset(unified_df.columns):
        return pd.DataFrame(columns=AMR_DRIVER_CATEGORY_COLUMNS)

    include_set = {str(value).strip().lower() for value in include_decisions}
    work = unified_df.copy()
    work["unified_decision"] = work["unified_decision"].astype(str).str.strip().str.lower()
    work = work.loc[work["unified_decision"].isin(include_set)].copy()
    if work.empty:
        return pd.DataFrame(columns=AMR_DRIVER_CATEGORY_COLUMNS)

    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["pathogen", "antibiotic", "year"])
    if work.empty:
        return pd.DataFrame(columns=AMR_DRIVER_CATEGORY_COLUMNS)

    work["year"] = work["year"].astype(int)
    work["amr_primary_category"] = work[category_col].where(work[category_col].notna(), WEAK_SIGNAL_CATEGORY)
    work["amr_primary_category"] = (
        work["amr_primary_category"].astype(str).str.strip().replace({"": WEAK_SIGNAL_CATEGORY})
    )

    return (
        work.groupby(["pathogen", "antibiotic", "year", "amr_primary_category"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["pathogen", "antibiotic", "year", "amr_primary_category"])
        .reset_index(drop=True)
    )


def summarize_si_change_drivers(
    interesting_df: pd.DataFrame,
    unified_df: pd.DataFrame,
    *,
    fit_end_year: int,
    include_decisions: tuple[str, ...] | list[str] | set[str] = ("yes", "uncertain"),
    lookback_years: int = 5,
    category_col: str = "amr_primary_category",
) -> pd.DataFrame:
    """Summarize recent LLM category mix for ranked SI-change pairs."""
    required = {
        "pathogen",
        "antibiotic",
        "year",
        "unified_decision",
        category_col,
        "amr_pair_centrality",
        "amr_needs_manual_review",
    }
    if interesting_df.empty or unified_df.empty or not required.issubset(unified_df.columns):
        return pd.DataFrame(columns=AMR_DRIVER_SUMMARY_COLUMNS)

    include_set = {str(value).strip().lower() for value in include_decisions}
    lookback_years = max(int(lookback_years), 1)
    recent_start = int(fit_end_year) - lookback_years + 1
    previous_start = int(fit_end_year) - (2 * lookback_years) + 1
    previous_end = recent_start - 1

    work = unified_df.copy()
    work["unified_decision"] = work["unified_decision"].astype(str).str.strip().str.lower()
    work = work.loc[work["unified_decision"].isin(include_set)].copy()
    if work.empty:
        return pd.DataFrame(columns=AMR_DRIVER_SUMMARY_COLUMNS)

    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["pathogen", "antibiotic", "year"])
    if work.empty:
        return pd.DataFrame(columns=AMR_DRIVER_SUMMARY_COLUMNS)

    work["year"] = work["year"].astype(int)
    work["amr_primary_category"] = work[category_col].where(work[category_col].notna(), WEAK_SIGNAL_CATEGORY)
    work["amr_primary_category"] = (
        work["amr_primary_category"].astype(str).str.strip().replace({"": WEAK_SIGNAL_CATEGORY})
    )
    work["amr_pair_centrality"] = work["amr_pair_centrality"].astype(str).str.strip().str.lower()
    manual = work["amr_needs_manual_review"]
    if manual.dtype == object:
        work["manual_review_bool"] = manual.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    else:
        work["manual_review_bool"] = manual.fillna(False).astype(bool)

    pairs = interesting_df[["pathogen", "antibiotic"]].drop_duplicates()
    work = work.merge(pairs, on=["pathogen", "antibiotic"], how="inner")
    if work.empty:
        return pd.DataFrame(columns=AMR_DRIVER_SUMMARY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        pair_df = work.loc[(work["pathogen"] == pair.pathogen) & (work["antibiotic"] == pair.antibiotic)]
        recent = pair_df.loc[(pair_df["year"] >= recent_start) & (pair_df["year"] <= int(fit_end_year))]
        previous = pair_df.loc[(pair_df["year"] >= previous_start) & (pair_df["year"] <= previous_end)]

        recent_counts = recent["amr_primary_category"].value_counts()
        previous_counts = previous["amr_primary_category"].value_counts()
        all_categories = sorted(set(recent_counts.index).union(previous_counts.index))
        changes = {
            category: int(recent_counts.get(category, 0) - previous_counts.get(category, 0))
            for category in all_categories
        }

        if recent_counts.empty:
            dominant_category = None
            dominant_count = 0
        else:
            dominant_category = sorted(
                ((str(category), int(count)) for category, count in recent_counts.items()),
                key=lambda item: (-item[1], item[0]),
            )[0][0]
            dominant_count = int(recent_counts[dominant_category])

        if changes:
            fastest_category, fastest_change = sorted(changes.items(), key=lambda item: (-item[1], item[0]))[0]
        else:
            fastest_category, fastest_change = None, 0

        recent_total = int(len(recent))
        weak_count = int((recent["amr_primary_category"] == WEAK_SIGNAL_CATEGORY).sum()) if recent_total else 0
        centrality_counts = recent["amr_pair_centrality"].value_counts()
        rows.append(
            {
                "pathogen": pair.pathogen,
                "antibiotic": pair.antibiotic,
                "driver_recent_total": recent_total,
                "driver_previous_total": int(len(previous)),
                "driver_count_change": recent_total - int(len(previous)),
                "driver_dominant_category": dominant_category,
                "driver_dominant_category_count": dominant_count,
                "driver_fastest_growing_category": fastest_category,
                "driver_fastest_growing_category_change": int(fastest_change),
                "driver_weak_signal_share": weak_count / recent_total if recent_total else 0.0,
                "driver_manual_review_share": float(recent["manual_review_bool"].mean()) if recent_total else 0.0,
                "driver_central_count": int(centrality_counts.get("central", 0)),
                "driver_partial_count": int(centrality_counts.get("partial", 0)),
                "driver_incidental_count": int(centrality_counts.get("incidental", 0)),
            }
        )

    return pd.DataFrame(rows, columns=AMR_DRIVER_SUMMARY_COLUMNS)


def _build_pair_lookup(si_df: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Precompute yearly pair tables once to avoid repeated full-frame filtering."""
    required = {
        "pathogen",
        "antibiotic",
        "year",
        "count",
        "bg_pathogen_n",
        "bg_antibiotic_n",
        "expected_count_global",
        "expected_log_count_global",
        "SI",
    }
    missing = required - set(si_df.columns)
    if missing:
        raise KeyError(f"si_df missing required yearly SI columns: {sorted(missing)}")

    grouped = (
        si_df[list(required)]
        .groupby(["pathogen", "antibiotic", "year"], as_index=False)
        .agg(
            {
                "count": "sum",
                "bg_pathogen_n": "max",
                "bg_antibiotic_n": "max",
                "expected_count_global": "sum",
                "expected_log_count_global": "mean",
                "SI": "mean",
            }
        )
        .sort_values(["pathogen", "antibiotic", "year"])
        .reset_index(drop=True)
    )
    grouped["log_count"] = _log_transform(grouped["count"].astype(float))

    out: dict[tuple[str, str], pd.DataFrame] = {}
    for (pathogen, antibiotic), frame in grouped.groupby(["pathogen", "antibiotic"], sort=False):
        out[(str(pathogen), str(antibiotic))] = frame.reset_index(drop=True)
    return out


def _loess_predict(
    train_years: np.ndarray,
    train_values: np.ndarray,
    query_years: np.ndarray,
    *,
    frac: float,
    degree: int,
) -> np.ndarray:
    train_years = np.asarray(train_years, dtype=float)
    train_values = np.asarray(train_values, dtype=float)
    query_years = np.asarray(query_years, dtype=float)
    if train_years.size == 0:
        raise ValueError("Cannot evaluate LOESS with no training points.")
    if train_years.size == 1:
        return np.full_like(query_years, float(train_values[0]), dtype=float)
    if degree != 1:
        raise ValueError("Only linear LOESS is supported; set loess_degree=1.")
    if loess is None:
        coeffs = np.polyfit(train_years, train_values, deg=1)
        return np.asarray(np.polyval(coeffs, query_years), dtype=float)
    model = loess(
        train_years,
        train_values,
        span=float(frac),
        degree=int(degree),
        family="gaussian",
        normalize=False,
    )
    model.fit()
    return np.asarray(model.predict(query_years).values, dtype=float)


def _fit_loess_log_count_curve(
    years: np.ndarray,
    log_counts: np.ndarray,
    *,
    loess_frac: float,
    loess_degree: int,
    tail_damping_years: Any = _AUTO_TAIL_DAMPING,
) -> dict[str, Any]:
    years = np.asarray(years, dtype=float)
    log_counts = np.asarray(log_counts, dtype=float)
    if years.size == 0:
        raise ValueError("Cannot fit LOESS with no training points.")
    if years.size == 1:
        level = float(log_counts[0])
        return {
            "frac": float(loess_frac),
            "degree": int(loess_degree),
            "train_fitted_log_count": np.full_like(log_counts, level, dtype=float),
            "train_residuals": np.zeros_like(log_counts, dtype=float),
            "residual_std": 0.0,
            "constant_level": level,
            "left_slope": 0.0,
            "right_slope": 0.0,
            "tail_damping_years": None,
            "year_min": float(years[0]),
            "year_max": float(years[0]),
            "left_value": level,
            "right_value": level,
            "train_years": years,
            "train_log_counts": log_counts,
            "fit_method": "constant",
            "linear_slope": None,
            "linear_intercept": None,
            "poly_coeffs": None,
        }

    if int(loess_degree) != 1:
        raise ValueError("Only linear LOESS is supported; set loess_degree=1.")

    try:
        fitted_log_counts = _loess_predict(years, log_counts, years, frac=loess_frac, degree=1)
        fitted_degree = 1
        fit_method = "loess_degree_1"
        linear_slope = None
        linear_intercept = None
        poly_coeffs = None
    except ValueError:
        slope, intercept = np.polyfit(years, log_counts, deg=1)
        fitted_log_counts = intercept + slope * years
        fitted_degree = 1
        fit_method = "linear_regression"
        linear_slope = float(slope)
        linear_intercept = float(intercept)
        poly_coeffs = None

    residuals = log_counts - fitted_log_counts
    residual_std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    if residual_std < 1e-8 and log_counts.size > 1:
        residual_std = float(max(1e-6, 0.25 * np.std(np.diff(log_counts), ddof=1)))

    left_slope = float((fitted_log_counts[1] - fitted_log_counts[0]) / (years[1] - years[0]))
    right_slope = float((fitted_log_counts[-1] - fitted_log_counts[-2]) / (years[-1] - years[-2]))
    if tail_damping_years is _AUTO_TAIL_DAMPING:
        span = float(max(years[-1] - years[0], 1.0))
        tail_damping_years = float(max(2.0, 0.10 * span))
    elif tail_damping_years is None:
        tail_damping_years = None
    else:
        tail_damping_years = float(max(tail_damping_years, 1e-6))

    return {
        "frac": float(loess_frac),
        "degree": int(fitted_degree),
        "train_fitted_log_count": fitted_log_counts,
        "train_residuals": residuals,
        "residual_std": residual_std,
        "constant_level": None,
        "left_slope": left_slope,
        "right_slope": right_slope,
        "tail_damping_years": tail_damping_years,
        "year_min": float(years[0]),
        "year_max": float(years[-1]),
        "left_value": float(fitted_log_counts[0]),
        "right_value": float(fitted_log_counts[-1]),
        "train_years": years,
        "train_log_counts": log_counts,
        "fit_method": fit_method,
        "linear_slope": linear_slope,
        "linear_intercept": linear_intercept,
        "poly_coeffs": poly_coeffs.tolist() if poly_coeffs is not None else None,
    }


def _damped_linear_extrapolation(
    base_value: float,
    slope: float,
    delta_years: np.ndarray,
    damping_years: float | None,
) -> np.ndarray:
    delta_years = np.asarray(delta_years, dtype=float)
    if damping_years is None:
        return base_value + slope * delta_years
    damping_years = float(max(damping_years, 1e-6))
    signed_decay = np.sign(delta_years) * (1.0 - np.exp(-np.abs(delta_years) / damping_years))
    return base_value + slope * damping_years * signed_decay


def _predict_loess_log_count_with_linear_tails(loess_fit: dict[str, Any], years: np.ndarray) -> np.ndarray:
    years = np.asarray(years, dtype=float)
    if loess_fit["constant_level"] is not None:
        return np.full_like(years, loess_fit["constant_level"], dtype=float)
    if loess_fit.get("fit_method") == "linear_regression":
        return loess_fit["linear_intercept"] + loess_fit["linear_slope"] * years

    pred = np.empty_like(years, dtype=float)
    inside = (years >= loess_fit["year_min"]) & (years <= loess_fit["year_max"])
    left = years < loess_fit["year_min"]
    right = years > loess_fit["year_max"]
    pred[inside] = _loess_predict(
        loess_fit["train_years"],
        loess_fit["train_log_counts"],
        years[inside],
        frac=loess_fit["frac"],
        degree=loess_fit["degree"],
    )
    pred[left] = _damped_linear_extrapolation(
        loess_fit["left_value"],
        loess_fit["left_slope"],
        years[left] - loess_fit["year_min"],
        loess_fit.get("tail_damping_years", None),
    )
    pred[right] = _damped_linear_extrapolation(
        loess_fit["right_value"],
        loess_fit["right_slope"],
        years[right] - loess_fit["year_max"],
        loess_fit.get("tail_damping_years", None),
    )
    return pred


def fit_predict_yearly_si_loess(
    si_df: pd.DataFrame,
    pathogen: str,
    antibiotic: str,
    fit_start_year: int,
    fit_end_year: int,
    num_additional_years: int = 5,
    *,
    min_train_points: int = 3,
    loess_frac: float = 0.6,
    loess_degree: int = 1,
    tail_damping_years: Any = _AUTO_TAIL_DAMPING,
    pair_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fit a smooth yearly count trajectory and derive yearly SI forecasts."""
    if fit_end_year < fit_start_year:
        raise ValueError("fit_end_year must be >= fit_start_year.")
    if num_additional_years < 0:
        raise ValueError("num_additional_years must be >= 0.")

    if pair_df is None:
        pair_df = build_pair_si_yearly_series(si_df, pathogen, antibiotic)
    else:
        pair_df = pair_df.copy().sort_values("year").reset_index(drop=True)
        if pair_df.empty:
            raise ValueError(f"No yearly SI data found for pathogen={pathogen!r}, antibiotic={antibiotic!r}.")
        if "log_count" not in pair_df.columns:
            pair_df["log_count"] = _log_transform(pair_df["count"].astype(float))
    latest_available_year = int(pair_df["year"].max())
    train_df = pair_df.loc[pair_df["year"].between(int(fit_start_year), int(fit_end_year))].copy()
    if train_df["year"].nunique() < min_train_points:
        raise ValueError(
            f"Need at least {min_train_points} observed years between {fit_start_year} and {fit_end_year}; "
            f"found {train_df['year'].nunique()}."
        )

    loess_fit = _fit_loess_log_count_curve(
        train_df["year"].to_numpy(dtype=float),
        train_df["log_count"].to_numpy(dtype=float),
        loess_frac=loess_frac,
        loess_degree=loess_degree,
        tail_damping_years=tail_damping_years,
    )

    prediction_end_year = latest_available_year + int(num_additional_years)
    model_plot_start_year = int(pair_df["year"].min())
    pred_years = np.arange(model_plot_start_year, prediction_end_year + 1, dtype=float)

    pred_df = pd.DataFrame({"year": pred_years.astype(int)})
    pred_df["expected_log_count_baseline"] = np.interp(
        pred_years,
        pair_df["year"].astype(float).to_numpy(),
        pair_df["expected_log_count_global"].astype(float).to_numpy(),
        left=float(pair_df["expected_log_count_global"].iloc[0]),
        right=float(pair_df["expected_log_count_global"].iloc[-1]),
    )
    pred_df["expected_count_baseline"] = np.clip(_inverse_log_transform(pred_df["expected_log_count_baseline"]), 0.0, None)

    pred_log_count = _predict_loess_log_count_with_linear_tails(loess_fit, pred_years)
    pred_df["predicted_log_count_loess"] = pred_log_count
    pred_df["predicted_log_count_std"] = float(loess_fit["residual_std"])
    pred_df["predicted_log_count_loess_lower_1std"] = pred_df["predicted_log_count_loess"] - pred_df["predicted_log_count_std"]
    pred_df["predicted_log_count_loess_upper_1std"] = pred_df["predicted_log_count_loess"] + pred_df["predicted_log_count_std"]
    pred_df["predicted_count_loess"] = np.clip(_inverse_log_transform(pred_df["predicted_log_count_loess"]), 0.0, None)
    pred_df["predicted_count_loess_lower_1std"] = np.clip(_inverse_log_transform(pred_df["predicted_log_count_loess_lower_1std"]), 0.0, None)
    pred_df["predicted_count_loess_upper_1std"] = np.clip(_inverse_log_transform(pred_df["predicted_log_count_loess_upper_1std"]), 0.0, None)
    pred_df["predicted_SI"] = pred_df["predicted_log_count_loess"] - pred_df["expected_log_count_baseline"]
    pred_df["predicted_std"] = float(loess_fit["residual_std"])
    pred_df["lower_1std"] = pred_df["predicted_SI"] - pred_df["predicted_std"]
    pred_df["upper_1std"] = pred_df["predicted_SI"] + pred_df["predicted_std"]

    observed_window_df = pair_df.loc[
        pair_df["year"] >= int(fit_start_year),
        ["year", "SI", "count", "log_count", "expected_count_global", "expected_log_count_global"],
    ].copy()
    observed_window_df = observed_window_df.rename(
        columns={"SI": "observed_SI", "count": "observed_count", "log_count": "observed_log_count"}
    )
    detailed_comparison_df = observed_window_df.merge(pred_df, on="year", how="left")
    detailed_comparison_df["prediction_error"] = detailed_comparison_df["observed_SI"] - detailed_comparison_df["predicted_SI"]
    detailed_comparison_df["abs_error"] = detailed_comparison_df["prediction_error"].abs()
    detailed_comparison_df["within_1std"] = (
        (detailed_comparison_df["observed_SI"] >= detailed_comparison_df["lower_1std"])
        & (detailed_comparison_df["observed_SI"] <= detailed_comparison_df["upper_1std"])
    )
    detailed_comparison_df["count_prediction_error_loess"] = detailed_comparison_df["observed_count"] - detailed_comparison_df["predicted_count_loess"]
    detailed_comparison_df["count_abs_error_loess"] = detailed_comparison_df["count_prediction_error_loess"].abs()

    compact = detailed_comparison_df[["year"]].copy()
    compact["observed_count"] = np.rint(detailed_comparison_df["observed_count"]).astype(int)
    compact["predicted_count"] = np.rint(detailed_comparison_df["predicted_count_loess"]).astype(int)
    compact["diff"] = compact["observed_count"] - compact["predicted_count"]
    compact["pct_error"] = np.where(
        compact["observed_count"] != 0,
        100.0 * compact["diff"] / compact["observed_count"],
        np.nan,
    )

    return {
        "pathogen": pathogen,
        "antibiotic": antibiotic,
        "fit_start_year": int(fit_start_year),
        "fit_end_year": int(fit_end_year),
        "latest_available_year": latest_available_year,
        "num_additional_years": int(num_additional_years),
        "loess_degree": int(loess_fit["degree"]),
        "model_plot_start_year": model_plot_start_year,
        "pair_df": pair_df,
        "train_df": train_df,
        "pred_df": pred_df,
        "comparison_df": compact,
        "detailed_comparison_df": detailed_comparison_df,
        "loess_fit": loess_fit,
    }


def rank_yearly_pairs_by_future_si_shift(
    si_df: pd.DataFrame,
    *,
    fit_start_year: int = 1970,
    fit_end_year: int | None = None,
    num_future_years: int = 20,
    reference_year: int | None = None,
    min_train_points: int = 3,
    loess_frac: float = 0.6,
    loess_degree: int = 1,
    tail_damping_years: Any = _AUTO_TAIL_DAMPING,
    save_path: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Rank yearly pathogen-antibiotic pairs by current vs future predicted SI."""
    required_cols = {"pathogen", "antibiotic", "year", "SI"}
    missing = required_cols - set(si_df.columns)
    if missing:
        raise KeyError(f"si_df is missing required yearly SI columns: {sorted(missing)}")
    if num_future_years < 0:
        raise ValueError("num_future_years must be >= 0.")

    if fit_end_year is None:
        fit_end_year = int(pd.to_numeric(si_df["year"], errors="coerce").max())
    if reference_year is None:
        reference_year = int(fit_end_year)
    if fit_end_year < fit_start_year:
        raise ValueError("fit_end_year must be >= fit_start_year.")

    reference_year = int(reference_year)
    fit_end_year = int(fit_end_year)
    target_year = int(reference_year + int(num_future_years))
    pair_lookup = _build_pair_lookup(si_df)
    pair_index_df = (
        pd.DataFrame(
            [{"pathogen": k[0], "antibiotic": k[1]} for k in pair_lookup.keys()]
        )
        .sort_values(["pathogen", "antibiotic"])
        .reset_index(drop=True)
    )
    total_pairs = int(len(pair_index_df))

    def _safe_pct_error(error: pd.Series, observed: pd.Series) -> pd.Series:
        observed = pd.to_numeric(observed, errors="coerce")
        error = pd.to_numeric(error, errors="coerce")
        denom = observed.abs()
        return pd.Series(np.where(denom > 0, 100.0 * error / denom, np.nan), index=error.index, dtype=float)

    rows: list[dict[str, Any]] = []
    for idx, pair in enumerate(pair_index_df.itertuples(index=False), start=1):
        pathogen = str(pair.pathogen)
        antibiotic = str(pair.antibiotic)
        if verbose:
            print(f"[{idx}/{total_pairs}] ranking {pathogen} vs {antibiotic}", flush=True)
        try:
            pair_df = pair_lookup[(pathogen, antibiotic)].copy()
            latest_available_year = int(pair_df["year"].max())
            horizon_years = max(int(num_future_years), target_year - latest_available_year, 0)
            result = fit_predict_yearly_si_loess(
                si_df=si_df,
                pathogen=pathogen,
                antibiotic=antibiotic,
                fit_start_year=fit_start_year,
                fit_end_year=fit_end_year,
                num_additional_years=horizon_years,
                min_train_points=min_train_points,
                loess_frac=loess_frac,
                loess_degree=loess_degree,
                tail_damping_years=tail_damping_years,
                pair_df=pair_df,
            )
            pred_df = result["pred_df"]
            current_row = pred_df.loc[pred_df["year"] == reference_year]
            future_row = pred_df.loc[pred_df["year"] == target_year]
            if current_row.empty:
                raise ValueError(f"Missing current-year prediction for reference_year={reference_year}.")
            if future_row.empty:
                raise ValueError(f"Missing future-year prediction for target_year={target_year}.")
            current_row = current_row.iloc[0]
            future_row = future_row.iloc[0]

            observed_current_df = pair_df.loc[pair_df["year"] == reference_year]
            current_source = "observed" if not observed_current_df.empty else "predicted"
            current_count = (
                float(observed_current_df["count"].iloc[0]) if not observed_current_df.empty and "count" in observed_current_df.columns else np.nan
            )
            holdout_df = result["detailed_comparison_df"].loc[result["detailed_comparison_df"]["year"] > fit_end_year].copy()
            holdout_df["si_pct_error"] = _safe_pct_error(holdout_df["prediction_error"], holdout_df["observed_SI"])
            n_holdout_years = int(len(holdout_df))
            latest_holdout_row = holdout_df.sort_values("year").tail(1)
            if latest_holdout_row.empty:
                latest_holdout_year = np.nan
                latest_observed_si = np.nan
                latest_predicted_si = np.nan
                latest_si_error = np.nan
                latest_si_abs_error = np.nan
                latest_si_pct_error = np.nan
                post_fit_si_error_mean = np.nan
                post_fit_si_abs_error_mean = np.nan
                post_fit_si_pct_error_mean = np.nan
                post_fit_si_abs_pct_error_mean = np.nan
            else:
                latest = latest_holdout_row.iloc[0]
                latest_holdout_year = int(latest["year"])
                latest_observed_si = float(latest["observed_SI"])
                latest_predicted_si = float(latest["predicted_SI"])
                latest_si_error = float(latest["prediction_error"])
                latest_si_abs_error = float(latest["abs_error"])
                latest_si_pct_error = float(latest["si_pct_error"])
                post_fit_si_error_mean = float(holdout_df["prediction_error"].mean())
                post_fit_si_abs_error_mean = float(holdout_df["abs_error"].mean())
                post_fit_si_pct_error_mean = float(holdout_df["si_pct_error"].mean())
                post_fit_si_abs_pct_error_mean = float(holdout_df["si_pct_error"].abs().mean())

            rows.append(
                {
                    "pathogen": pathogen,
                    "antibiotic": antibiotic,
                    "reference_year": reference_year,
                    "future_year": target_year,
                    "num_future_years": int(num_future_years),
                    "fit_start_year": fit_start_year,
                    "fit_end_year": fit_end_year,
                    "latest_available_year": latest_available_year,
                    "n_pair_years": int(pair_df["year"].nunique()),
                    "current_si_source": current_source,
                    "current_count": current_count,
                    "observed_current_SI": float(observed_current_df["SI"].iloc[0]) if not observed_current_df.empty else np.nan,
                    "current_predicted_SI": float(current_row["predicted_SI"]),
                    "current_predicted_count": float(current_row["predicted_count_loess"]),
                    "future_predicted_SI": float(future_row["predicted_SI"]),
                    "future_predicted_count": float(future_row["predicted_count_loess"]),
                    "si_change": float(future_row["predicted_SI"] - current_row["predicted_SI"]),
                    "count_change": float(future_row["predicted_count_loess"] - current_row["predicted_count_loess"]),
                    "n_post_fit_observed_years": n_holdout_years,
                    "latest_post_fit_observed_year": latest_holdout_year,
                    "latest_post_fit_observed_SI": latest_observed_si,
                    "latest_post_fit_predicted_SI": latest_predicted_si,
                    "latest_post_fit_si_error": latest_si_error,
                    "latest_post_fit_si_abs_error": latest_si_abs_error,
                    "latest_post_fit_si_pct_error": latest_si_pct_error,
                    "post_fit_si_error_mean": post_fit_si_error_mean,
                    "post_fit_si_abs_error_mean": post_fit_si_abs_error_mean,
                    "post_fit_si_pct_error_mean": post_fit_si_pct_error_mean,
                    "post_fit_si_abs_pct_error_mean": post_fit_si_abs_pct_error_mean,
                    "loess_frac": float(result["loess_fit"]["frac"]),
                    "loess_degree": int(result["loess_fit"]["degree"]),
                    "tail_damping_years": result["loess_fit"]["tail_damping_years"],
                    "status": "ok",
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "pathogen": pathogen,
                    "antibiotic": antibiotic,
                    "reference_year": reference_year,
                    "future_year": target_year,
                    "num_future_years": int(num_future_years),
                    "fit_start_year": fit_start_year,
                    "fit_end_year": fit_end_year,
                    "latest_available_year": np.nan,
                    "n_pair_years": np.nan,
                    "current_si_source": None,
                    "observed_current_SI": np.nan,
                    "current_predicted_SI": np.nan,
                    "current_predicted_count": np.nan,
                    "future_predicted_SI": np.nan,
                    "future_predicted_count": np.nan,
                    "si_change": np.nan,
                    "count_change": np.nan,
                    "n_post_fit_observed_years": np.nan,
                    "latest_post_fit_observed_year": np.nan,
                    "latest_post_fit_observed_SI": np.nan,
                    "latest_post_fit_predicted_SI": np.nan,
                    "latest_post_fit_si_error": np.nan,
                    "latest_post_fit_si_abs_error": np.nan,
                    "latest_post_fit_si_pct_error": np.nan,
                    "post_fit_si_error_mean": np.nan,
                    "post_fit_si_abs_error_mean": np.nan,
                    "post_fit_si_pct_error_mean": np.nan,
                    "post_fit_si_abs_pct_error_mean": np.nan,
                    "loess_frac": float(loess_frac),
                    "loess_degree": int(loess_degree),
                    "tail_damping_years": np.nan if tail_damping_years in (_AUTO_TAIL_DAMPING, None) else float(tail_damping_years),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    ranking_df = pd.DataFrame(rows)
    ok_mask = ranking_df["status"].eq("ok")
    ranking_df.loc[ok_mask, "current_rank"] = ranking_df.loc[ok_mask, "current_predicted_SI"].rank(method="min", ascending=False).astype(int)
    ranking_df.loc[ok_mask, "future_rank"] = ranking_df.loc[ok_mask, "future_predicted_SI"].rank(method="min", ascending=False).astype(int)
    ranking_df["rank_change"] = ranking_df["current_rank"] - ranking_df["future_rank"]
    ranking_df["abs_rank_change"] = ranking_df["rank_change"].abs()
    ranking_df = ranking_df.sort_values(
        ["status", "abs_rank_change", "future_predicted_SI", "current_predicted_SI"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    if save_path is None:
        save_path = f"si_yearly_rank_shift_fit_{fit_start_year}_{fit_end_year}_ref_{reference_year}_future_{target_year}.parquet"
        atomic_write_parquet(ranking_df, save_path, index=False)
    return ranking_df


def select_interesting_yearly_rank_cases(
    rank_df: pd.DataFrame,
    *,
    min_current_predicted_count: float = 50.0,
    min_future_predicted_count: float = 50.0,
    min_si_change: float = 0.75,
    min_post_fit_observed_years: int = 3,
    max_post_fit_si_abs_error_mean: float = 0.5,
    max_post_fit_si_abs_pct_error_mean: float = 50.0,
    require_post_fit_validation: bool = True,
    fallback_top_n: int = 50,
    max_count_for_scoring: float = 1e4,
    max_future_predicted_count: float | None = 1e5,
    max_future_to_current_ratio: float | None = 100.0,
    sort_by: str = "interest_score",
) -> pd.DataFrame:
    """Filter and rank interesting yearly SI cases from rank output."""
    required_cols = {"status", "current_predicted_count", "future_predicted_count", "si_change"}
    missing = required_cols - set(rank_df.columns)
    if missing:
        raise KeyError(f"rank_df is missing required columns: {sorted(missing)}")

    work = rank_df.loc[rank_df["status"].eq("ok")].copy()
    for col in ["post_fit_si_abs_error_mean", "post_fit_si_abs_pct_error_mean", "n_post_fit_observed_years"]:
        if col not in work.columns:
            work[col] = np.nan

    work["passes_current_count"] = pd.to_numeric(work["current_predicted_count"], errors="coerce") >= float(min_current_predicted_count)
    work["passes_future_count"] = pd.to_numeric(work["future_predicted_count"], errors="coerce") >= float(min_future_predicted_count)
    current_count = pd.to_numeric(work["current_predicted_count"], errors="coerce")
    future_count = pd.to_numeric(work["future_predicted_count"], errors="coerce")
    work["future_to_current_ratio"] = future_count / current_count.where(current_count > 0)
    work["passes_si_change"] = pd.to_numeric(work["si_change"], errors="coerce") >= float(min_si_change)
    work["passes_future_count_ceiling"] = True if max_future_predicted_count is None else (future_count <= float(max_future_predicted_count))
    work["passes_future_growth_ratio"] = True if max_future_to_current_ratio is None else (work["future_to_current_ratio"] <= float(max_future_to_current_ratio))

    post_fit_years = pd.to_numeric(work["n_post_fit_observed_years"], errors="coerce")
    post_fit_abs_error = pd.to_numeric(work["post_fit_si_abs_error_mean"], errors="coerce")
    post_fit_abs_pct_error = pd.to_numeric(work["post_fit_si_abs_pct_error_mean"], errors="coerce")
    has_post_fit_validation = post_fit_years.notna() & post_fit_abs_error.notna() & post_fit_abs_pct_error.notna()
    work["has_post_fit_validation"] = has_post_fit_validation
    work["passes_post_fit_years"] = post_fit_years >= int(min_post_fit_observed_years)
    work["passes_post_fit_abs_error"] = post_fit_abs_error <= float(max_post_fit_si_abs_error_mean)
    work["passes_post_fit_abs_pct_error"] = post_fit_abs_pct_error <= float(max_post_fit_si_abs_pct_error_mean)
    if require_post_fit_validation:
        work["passes_validation"] = (~work["has_post_fit_validation"]) | (
            work["passes_post_fit_years"] & work["passes_post_fit_abs_error"] & work["passes_post_fit_abs_pct_error"]
        )
    else:
        work["passes_validation"] = (~work["has_post_fit_validation"]) | (
            work["passes_post_fit_abs_error"] & work["passes_post_fit_abs_pct_error"]
        )

    work["is_interesting_case"] = (
        work["passes_current_count"]
        & work["passes_future_count"]
        & work["passes_future_count_ceiling"]
        & work["passes_future_growth_ratio"]
        & work["passes_si_change"]
        & work["passes_validation"]
    )

    scale = np.log1p(np.clip(current_count, 0.0, float(max_count_for_scoring)))
    scale = scale / np.log1p(float(max_count_for_scoring))
    reliability_penalty = np.clip(post_fit_abs_pct_error.fillna(0.0) / max(float(max_post_fit_si_abs_pct_error_mean), 1e-6), 0.0, 1.0)
    work["reliability_penalty"] = reliability_penalty
    work["interest_score"] = (
        np.maximum(pd.to_numeric(work["si_change"], errors="coerce"), 0.0)
        * (0.35 + 0.65 * scale)
        * (1.0 - 0.50 * reliability_penalty)
    )

    selected = work.loc[work["is_interesting_case"]].copy()
    if selected.empty and fallback_top_n > 0:
        selected = work.sort_values(["interest_score", "future_predicted_SI"], ascending=[False, False]).head(int(fallback_top_n)).copy()
        selected["selection_mode"] = "fallback_top_n"
    else:
        selected["selection_mode"] = "strict_filter"

    if sort_by not in selected.columns:
        sort_by = "interest_score"
    selected = selected.sort_values([sort_by, "future_predicted_SI"], ascending=[False, False]).reset_index(drop=True)
    return selected


def plot_yearly_si_pair(
    si_df: pd.DataFrame,
    pathogen: str,
    antibiotic: str,
    fit_start_year: int,
    fit_end_year: int,
    num_additional_years: int = 20,
    *,
    loess_frac: float = 0.6,
    loess_degree: int = 1,
    tail_damping_years: Any = None,
) -> dict[str, Any]:
    """Plot observed/predicted SI and counts for one pathogen-antibiotic pair."""
    result = fit_predict_yearly_si_loess(
        si_df=si_df,
        pathogen=pathogen,
        antibiotic=antibiotic,
        fit_start_year=fit_start_year,
        fit_end_year=fit_end_year,
        num_additional_years=num_additional_years,
        loess_frac=loess_frac,
        loess_degree=loess_degree,
        tail_damping_years=tail_damping_years,
    )
    pair_df = result["pair_df"]
    pred_df = result["pred_df"]

    fig, ax1 = plt.subplots(figsize=(12, 7), facecolor="white")
    ax2 = ax1.twinx()
    ax1.set_facecolor("#fbfbf8")
    ax2.set_facecolor("none")
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_alpha(0.0)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    ax2.spines["top"].set_visible(False)

    count_plot_df = pair_df[["year", "count", "bg_pathogen_n", "bg_antibiotic_n"]].copy()
    count_plot_df["amr_count_plot"] = np.clip(count_plot_df["count"].astype(float), 1.0, None)
    count_plot_df["pathogen_count_plot"] = np.clip(count_plot_df["bg_pathogen_n"].astype(float), 1.0, None)
    count_plot_df["antibiotic_count_plot"] = np.clip(count_plot_df["bg_antibiotic_n"].astype(float), 1.0, None)
    bar_width = 0.85
    ax2.bar(
        count_plot_df["year"],
        count_plot_df["pathogen_count_plot"],
        color="#8fbf88",
        alpha=0.26,
        edgecolor="none",
        width=bar_width,
        label="Pathogen publication count",
        zorder=0,
    )
    ax2.bar(
        count_plot_df["year"],
        count_plot_df["antibiotic_count_plot"],
        color="#b9a7d1",
        alpha=0.34,
        edgecolor="none",
        width=bar_width,
        label="Antibiotic publication count",
        zorder=1,
    )
    ax2.bar(
        count_plot_df["year"],
        count_plot_df["amr_count_plot"],
        color="#9ecae1",
        alpha=0.68,
        edgecolor="none",
        width=bar_width,
        label="AMR pair publication count",
        zorder=2,
    )
    ax2.set_yscale("log")

    ax1.plot(
        pair_df["year"],
        pair_df["SI"],
        color="#111111",
        marker="o",
        lw=2.6,
        markersize=6.0,
        markerfacecolor="#f8fbff",
        markeredgecolor="#111111",
        markeredgewidth=1.6,
        label="Observed SI",
        zorder=4,
    )
    ax1.plot(pred_df["year"], pred_df["predicted_SI"], color="#d62828", lw=2.8, label="Predicted SI", zorder=5)
    ax1.fill_between(
        pred_df["year"],
        pred_df["lower_1std"],
        pred_df["upper_1std"],
        color="#f7b2b0",
        alpha=0.28,
        label="Predicted SI ±1 std",
        zorder=3,
    )
    ax1.axvline(int(fit_end_year), color="#6b7280", linestyle="--", linewidth=1.6, alpha=0.9, label="Fit end year")
    ax2.plot(
        pred_df["year"],
        np.clip(pred_df["predicted_count_loess"], 1.0, None),
        color="#4e8fb3",
        lw=2.5,
        label="Predicted AMR pair count",
        zorder=3,
    )

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Scientific interest (SI)")
    ax2.set_ylabel("Publication counts (log scale)", color="#4e6475")
    ax2.tick_params(axis="y", labelcolor="#4e6475", colors="#4e6475")
    ax1.set_title(f"{pathogen} vs {antibiotic} | fit {fit_start_year}-{fit_end_year}")
    ax1.grid(True, axis="y", color="#c7c7c7", alpha=0.45, linewidth=0.8)
    ax1.grid(False, axis="x")
    ax2.grid(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=2.4,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))

    result["figure"] = fig
    result["ax_si"] = ax1
    result["ax_count"] = ax2
    return result


def plot_yearly_category_driver_counts(
    category_counts_df: pd.DataFrame,
    pathogen: str,
    antibiotic: str,
    *,
    fit_end_year: int | None = None,
) -> dict[str, Any]:
    """Plot yearly LLM category counts for one pathogen-antibiotic pair."""
    required = set(AMR_DRIVER_CATEGORY_COLUMNS)
    missing = required - set(category_counts_df.columns)
    if missing:
        raise KeyError(f"category_counts_df missing required columns: {sorted(missing)}")

    pair_df = category_counts_df.loc[
        (category_counts_df["pathogen"] == pathogen) & (category_counts_df["antibiotic"] == antibiotic)
    ].copy()
    if pair_df.empty:
        raise ValueError(f"No category counts found for pathogen={pathogen!r}, antibiotic={antibiotic!r}.")

    pivot = (
        pair_df.pivot_table(
            index="year",
            columns="amr_primary_category",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )
    ordered_categories = [category for category in AMR_DRIVER_CATEGORY_ORDER if category in pivot.columns]
    extra_categories = sorted(category for category in pivot.columns if category not in AMR_DRIVER_CATEGORY_COLORS)
    pivot = pivot[ordered_categories + extra_categories]
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = [
        AMR_DRIVER_CATEGORY_COLORS.get(category, fallback_colors[idx % len(fallback_colors)] if fallback_colors else "#999999")
        for idx, category in enumerate(pivot.columns)
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.85, color=colors)
    if fit_end_year is not None and not pivot.empty:
        years = list(pivot.index)
        if int(fit_end_year) in years:
            ax.axvline(
                years.index(int(fit_end_year)) + 0.5,
                color="grey",
                linestyle="--",
                linewidth=1.2,
                label="Fit end year",
            )
    ax.set_xlabel("Year")
    ax.set_ylabel("LLM-classified AMR publication count")
    ax.set_title(f"{pathogen} vs {antibiotic} | AMR category drivers")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return {"figure": fig, "ax": ax, "pivot": pivot}


__all__ = [
    "fit_global_pooled_si_model",
    "build_pair_si_yearly_series",
    "fit_predict_yearly_si_loess",
    "rank_yearly_pairs_by_future_si_shift",
    "select_interesting_yearly_rank_cases",
    "build_yearly_category_counts",
    "summarize_si_change_drivers",
    "plot_yearly_si_pair",
    "plot_yearly_category_driver_counts",
]
