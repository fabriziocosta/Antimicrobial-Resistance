"""Scientific-interest utilities for AMR interval analyses."""

from __future__ import annotations

import os
import re
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from skmisc.loess import loess
from sklearn.linear_model import Ridge

from .normalisation import BackgroundNormaliseConfig, normalise_by_background

_AUTO_TAIL_DAMPING = object()


def load_amr_background_df(
    *,
    alpha: float = 1.0,
    use_cached_norm: bool = True,
    save_cache: bool = True,
    cache_path: str = "amr_df_normalised.parquet",
    amr_path: str = "pmids_results_positive.parquet",
    pathogen_counts_path: str = "pathogen_tiab_counts.parquet",
    antibiotic_counts_path: str = "antibiotic_tiab_counts.parquet",
) -> pd.DataFrame:
    """Load interval AMR data and attach pathogen/antibiotic background counts."""
    if use_cached_norm and os.path.exists(cache_path):
        amr_background_df = pd.read_parquet(cache_path)
    else:
        amr_df = pd.read_parquet(amr_path)
        pathogen_counts_df = pd.read_parquet(pathogen_counts_path)
        antibiotic_counts_df = pd.read_parquet(antibiotic_counts_path)
        amr_background_df = normalise_by_background(
            amr_df,
            pathogen_year_df=pathogen_counts_df,
            antibiotic_year_df=antibiotic_counts_df,
            cfg=BackgroundNormaliseConfig(alpha=alpha),
        )
        if save_cache:
            amr_background_df.to_parquet(cache_path, index=False)

    amr_background_df = amr_background_df.copy()
    amr_background_df["start_year"] = pd.to_datetime(amr_background_df["start_date"]).dt.year.astype(int)
    amr_background_df["end_year"] = pd.to_datetime(amr_background_df["end_date"]).dt.year.astype(int)
    amr_background_df["mid_year"] = (amr_background_df["start_year"] + amr_background_df["end_year"]) / 2.0
    amr_background_df["log_count"] = np.log1p(pd.to_numeric(amr_background_df["count"], errors="coerce"))
    amr_background_df["log_bg_pathogen_n"] = np.log1p(pd.to_numeric(amr_background_df["bg_pathogen_n"], errors="coerce"))
    amr_background_df["log_bg_antibiotic_n"] = np.log1p(pd.to_numeric(amr_background_df["bg_antibiotic_n"], errors="coerce"))
    return amr_background_df.dropna(subset=["log_count", "log_bg_pathogen_n", "log_bg_antibiotic_n"])


def fit_global_pooled_si_model(
    amr_background_df: pd.DataFrame,
    *,
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, Ridge, list[str]]:
    """Fit the pooled expected-count baseline and define SI as the log residual."""
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
    work["expected_count_global"] = np.clip(np.expm1(work["expected_log_count_global"]), 0.0, None)
    work["SI"] = work["log_count"] - work["expected_log_count_global"]
    return work, model, design_df.columns.tolist()


def build_pair_si_interval_series(si_df: pd.DataFrame, pathogen: str, antibiotic: str) -> pd.DataFrame:
    """Collapse SI data to one row per pathogen-antibiotic interval."""
    pair_df = (
        si_df.loc[
            (si_df["pathogen"] == pathogen) & (si_df["antibiotic"] == antibiotic),
            [
                "pathogen",
                "antibiotic",
                "start_date",
                "end_date",
                "start_year",
                "end_year",
                "mid_year",
                "count",
                "bg_pathogen_n",
                "bg_antibiotic_n",
                "expected_count_global",
                "expected_log_count_global",
                "SI",
            ],
        ]
        .sort_values(["end_year", "start_year"])
        .reset_index(drop=True)
    )

    if pair_df.empty:
        raise ValueError(f"No SI data found for pathogen={pathogen!r}, antibiotic={antibiotic!r}.")

    pair_df = (
        pair_df.groupby(["pathogen", "antibiotic", "start_year", "end_year", "mid_year"], as_index=False)
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
        .sort_values("end_year")
        .reset_index(drop=True)
    )
    pair_df["log_count"] = np.log1p(pair_df["count"].astype(float))
    return pair_df


def build_pair_si_yearly_series(si_df: pd.DataFrame, pathogen: str, antibiotic: str) -> pd.DataFrame:
    """Collapse SI data to one row per pathogen-antibiotic year."""
    required_cols = {
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
    missing = required_cols - set(si_df.columns)
    if missing:
        raise KeyError(f"si_df is missing required yearly SI columns: {sorted(missing)}")

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
        .reset_index(drop=True)
    )

    if pair_df.empty:
        raise ValueError(f"No yearly SI data found for pathogen={pathogen!r}, antibiotic={antibiotic!r}.")

    pair_df = (
        pair_df.groupby(["pathogen", "antibiotic", "year"], as_index=False)
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
        .sort_values("year")
        .reset_index(drop=True)
    )
    pair_df["log_count"] = np.log1p(pair_df["count"].astype(float))
    return pair_df


def _interpolate_expected_log_count_curve(pair_df: pd.DataFrame, years: np.ndarray) -> np.ndarray:
    """Interpolate the pooled expected log-count baseline onto a yearly grid."""
    anchor_years = pair_df["end_year"].astype(int).to_numpy()
    anchor_vals = pair_df["expected_log_count_global"].astype(float).to_numpy()
    years = np.asarray(years, dtype=float)
    if anchor_years.size == 0:
        return np.zeros_like(years, dtype=float)
    if anchor_years.size == 1:
        return np.full_like(years, float(anchor_vals[0]), dtype=float)
    return np.interp(years, anchor_years.astype(float), anchor_vals, left=anchor_vals[0], right=anchor_vals[-1])


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
    if degree not in (1, 2):
        raise ValueError("loess_degree must be 1 or 2 for skmisc.loess.")

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
        constant_level = float(log_counts[0])
        return {
            "frac": float(loess_frac),
            "degree": int(loess_degree),
            "train_fitted_log_count": np.full_like(log_counts, constant_level, dtype=float),
            "train_residuals": np.zeros_like(log_counts, dtype=float),
            "residual_std": 0.0,
            "constant_level": constant_level,
            "left_slope": 0.0,
            "right_slope": 0.0,
            "tail_damping_years": None if tail_damping_years is None else 1.0,
            "year_min": float(years[0]),
            "year_max": float(years[0]),
            "left_value": constant_level,
            "right_value": constant_level,
            "train_years": years,
            "train_log_counts": log_counts,
            "fit_method": "constant",
            "linear_slope": None,
            "linear_intercept": None,
            "poly_coeffs": None,
        }

    try:
        fitted_log_counts = _loess_predict(years, log_counts, years, frac=loess_frac, degree=2)
        fitted_degree = 2
        fit_method = "loess_degree_2"
        linear_slope = None
        linear_intercept = None
        poly_coeffs = None
    except ValueError as exc:
        warnings.warn(
            f"Quadratic LOESS was numerically unstable ({exc}). Falling back to linear LOESS.",
            RuntimeWarning,
        )
        try:
            fitted_log_counts = _loess_predict(years, log_counts, years, frac=loess_frac, degree=1)
            fitted_degree = 1
            fit_method = "loess_degree_1"
            linear_slope = None
            linear_intercept = None
            poly_coeffs = None
        except ValueError as fallback_exc:
            warnings.warn(
                f"Linear LOESS was numerically unstable ({fallback_exc}). Falling back to quadratic polynomial fit.",
                RuntimeWarning,
            )
            try:
                poly_coeffs = np.polyfit(years, log_counts, deg=2)
                fitted_log_counts = np.polyval(poly_coeffs, years)
                fitted_degree = 2
                fit_method = "quadratic_regression"
                linear_slope = None
                linear_intercept = None
            except np.linalg.LinAlgError as poly_exc:
                warnings.warn(
                    f"Quadratic polynomial fit was numerically unstable ({poly_exc}). Falling back to ordinary linear regression.",
                    RuntimeWarning,
                )
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
        if log_counts.size > 2:
            residual_std = float(0.25 * np.std(np.diff(log_counts), ddof=1))
        else:
            residual_std = float(0.1 * abs(log_counts[-1] - log_counts[0]))

    left_slope = float((fitted_log_counts[1] - fitted_log_counts[0]) / (years[1] - years[0]))
    right_slope = float((fitted_log_counts[-1] - fitted_log_counts[-2]) / (years[-1] - years[-2]))
    if tail_damping_years is _AUTO_TAIL_DAMPING:
        tail_span_years = float(max(years[-1] - years[0], 1.0))
        tail_damping_years = float(max(2.0, 0.10 * tail_span_years))
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
    if loess_fit.get("fit_method") == "quadratic_regression":
        return np.polyval(np.asarray(loess_fit["poly_coeffs"], dtype=float), years)
    if loess_fit.get("fit_method") == "linear_regression":
        return loess_fit["linear_intercept"] + loess_fit["linear_slope"] * years

    pred = np.empty_like(years, dtype=float)
    inside_mask = (years >= loess_fit["year_min"]) & (years <= loess_fit["year_max"])
    left_mask = years < loess_fit["year_min"]
    right_mask = years > loess_fit["year_max"]
    pred[inside_mask] = _loess_predict(
        loess_fit["train_years"],
        loess_fit["train_log_counts"],
        years[inside_mask],
        frac=loess_fit["frac"],
        degree=loess_fit["degree"],
    )
    pred[left_mask] = _damped_linear_extrapolation(
        loess_fit["left_value"],
        loess_fit["left_slope"],
        years[left_mask] - loess_fit["year_min"],
        loess_fit.get("tail_damping_years", None),
    )
    pred[right_mask] = _damped_linear_extrapolation(
        loess_fit["right_value"],
        loess_fit["right_slope"],
        years[right_mask] - loess_fit["year_max"],
        loess_fit.get("tail_damping_years", None),
    )
    return pred


def fit_predict_si_loess(
    si_df: pd.DataFrame,
    pathogen: str,
    antibiotic: str,
    fit_start_year: int,
    fit_end_year: int,
    num_additional_years: int = 5,
    *,
    min_train_points: int = 3,
    loess_frac: float = 0.6,
    loess_degree: int = 2,
    tail_damping_years: Any = _AUTO_TAIL_DAMPING,
) -> dict[str, Any]:
    """Fit a smooth count trajectory and derive SI forecasts from it."""
    if fit_end_year < fit_start_year:
        raise ValueError("fit_end_year must be >= fit_start_year.")
    if num_additional_years < 0:
        raise ValueError("num_additional_years must be >= 0.")

    pair_df = build_pair_si_interval_series(si_df, pathogen, antibiotic)
    latest_available_year = int(pair_df["end_year"].max())

    train_df = pair_df.loc[pair_df["end_year"].between(int(fit_start_year), int(fit_end_year))].copy()
    if train_df["end_year"].nunique() < min_train_points:
        raise ValueError(
            f"Need at least {min_train_points} observed intervals between {fit_start_year} and {fit_end_year}; "
            f"found {train_df['end_year'].nunique()}."
        )

    loess_fit = _fit_loess_log_count_curve(
        train_df["end_year"].to_numpy(dtype=float),
        train_df["log_count"].to_numpy(dtype=float),
        loess_frac=loess_frac,
        loess_degree=loess_degree,
        tail_damping_years=tail_damping_years,
    )

    prediction_end_year = latest_available_year + int(num_additional_years)
    model_plot_start_year = int(pair_df["end_year"].min())
    pred_years = np.arange(model_plot_start_year, prediction_end_year + 1, dtype=float)

    pred_df = pd.DataFrame({"year": pred_years.astype(int)})
    pred_df["expected_log_count_baseline"] = _interpolate_expected_log_count_curve(pair_df, pred_years)
    pred_df["expected_count_baseline"] = np.clip(np.expm1(pred_df["expected_log_count_baseline"]), 0.0, None)

    pred_log_count = _predict_loess_log_count_with_linear_tails(loess_fit, pred_years)
    pred_df["predicted_log_count_loess"] = pred_log_count
    pred_df["predicted_log_count_std"] = float(loess_fit["residual_std"])
    pred_df["predicted_log_count_loess_lower_1std"] = pred_df["predicted_log_count_loess"] - pred_df["predicted_log_count_std"]
    pred_df["predicted_log_count_loess_upper_1std"] = pred_df["predicted_log_count_loess"] + pred_df["predicted_log_count_std"]
    pred_df["predicted_count_loess"] = np.clip(np.expm1(pred_df["predicted_log_count_loess"]), 0.0, None)
    pred_df["predicted_count_loess_lower_1std"] = np.clip(np.expm1(pred_df["predicted_log_count_loess_lower_1std"]), 0.0, None)
    pred_df["predicted_count_loess_upper_1std"] = np.clip(np.expm1(pred_df["predicted_log_count_loess_upper_1std"]), 0.0, None)

    pred_df["predicted_SI"] = pred_df["predicted_log_count_loess"] - pred_df["expected_log_count_baseline"]
    pred_df["predicted_std"] = float(loess_fit["residual_std"])
    pred_df["lower_1std"] = pred_df["predicted_SI"] - pred_df["predicted_std"]
    pred_df["upper_1std"] = pred_df["predicted_SI"] + pred_df["predicted_std"]

    observed_window_df = pair_df.loc[
        pair_df["end_year"] >= int(fit_start_year),
        ["end_year", "SI", "count", "log_count", "expected_count_global", "expected_log_count_global"],
    ].copy()
    observed_window_df = observed_window_df.rename(
        columns={
            "end_year": "year",
            "SI": "observed_SI",
            "count": "observed_count",
            "log_count": "observed_log_count",
        }
    )
    comparison_df = observed_window_df.merge(pred_df, on="year", how="left")
    comparison_df["prediction_error"] = comparison_df["observed_SI"] - comparison_df["predicted_SI"]
    comparison_df["abs_error"] = comparison_df["prediction_error"].abs()
    comparison_df["within_1std"] = (
        (comparison_df["observed_SI"] >= comparison_df["lower_1std"])
        & (comparison_df["observed_SI"] <= comparison_df["upper_1std"])
    )
    comparison_df["count_prediction_error_loess"] = comparison_df["observed_count"] - comparison_df["predicted_count_loess"]
    comparison_df["count_abs_error_loess"] = comparison_df["count_prediction_error_loess"].abs()
    preferred_cols = [
        "year",
        "observed_count",
        "predicted_count_loess",
    ]
    remaining_cols = [col for col in comparison_df.columns if col not in preferred_cols]
    comparison_df = comparison_df[preferred_cols + remaining_cols]

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
        "comparison_df": comparison_df,
        "loess_fit": loess_fit,
    }


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
    loess_degree: int = 2,
    tail_damping_years: Any = _AUTO_TAIL_DAMPING,
) -> dict[str, Any]:
    """Fit a smooth yearly count trajectory and derive yearly SI forecasts from it."""
    if fit_end_year < fit_start_year:
        raise ValueError("fit_end_year must be >= fit_start_year.")
    if num_additional_years < 0:
        raise ValueError("num_additional_years must be >= 0.")

    pair_df = build_pair_si_yearly_series(si_df, pathogen, antibiotic)
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
    pred_df["expected_count_baseline"] = np.clip(np.expm1(pred_df["expected_log_count_baseline"]), 0.0, None)

    pred_log_count = _predict_loess_log_count_with_linear_tails(loess_fit, pred_years)
    pred_df["predicted_log_count_loess"] = pred_log_count
    pred_df["predicted_log_count_std"] = float(loess_fit["residual_std"])
    pred_df["predicted_log_count_loess_lower_1std"] = pred_df["predicted_log_count_loess"] - pred_df["predicted_log_count_std"]
    pred_df["predicted_log_count_loess_upper_1std"] = pred_df["predicted_log_count_loess"] + pred_df["predicted_log_count_std"]
    pred_df["predicted_count_loess"] = np.clip(np.expm1(pred_df["predicted_log_count_loess"]), 0.0, None)
    pred_df["predicted_count_loess_lower_1std"] = np.clip(np.expm1(pred_df["predicted_log_count_loess_lower_1std"]), 0.0, None)
    pred_df["predicted_count_loess_upper_1std"] = np.clip(np.expm1(pred_df["predicted_log_count_loess_upper_1std"]), 0.0, None)

    pred_df["predicted_SI"] = pred_df["predicted_log_count_loess"] - pred_df["expected_log_count_baseline"]
    pred_df["predicted_std"] = float(loess_fit["residual_std"])
    pred_df["lower_1std"] = pred_df["predicted_SI"] - pred_df["predicted_std"]
    pred_df["upper_1std"] = pred_df["predicted_SI"] + pred_df["predicted_std"]

    observed_window_df = pair_df.loc[
        pair_df["year"] >= int(fit_start_year),
        ["year", "SI", "count", "log_count", "expected_count_global", "expected_log_count_global"],
    ].copy()
    observed_window_df = observed_window_df.rename(
        columns={
            "SI": "observed_SI",
            "count": "observed_count",
            "log_count": "observed_log_count",
        }
    )
    comparison_df = observed_window_df.merge(pred_df, on="year", how="left")
    comparison_df["prediction_error"] = comparison_df["observed_SI"] - comparison_df["predicted_SI"]
    comparison_df["abs_error"] = comparison_df["prediction_error"].abs()
    comparison_df["within_1std"] = (
        (comparison_df["observed_SI"] >= comparison_df["lower_1std"])
        & (comparison_df["observed_SI"] <= comparison_df["upper_1std"])
    )
    comparison_df["count_prediction_error_loess"] = comparison_df["observed_count"] - comparison_df["predicted_count_loess"]
    comparison_df["count_abs_error_loess"] = comparison_df["count_prediction_error_loess"].abs()

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
        "comparison_df": comparison_df,
        "loess_fit": loess_fit,
    }


def forecast_all_pair_si_loess(
    si_df: pd.DataFrame,
    fit_start_year: int = 1970,
    fit_end_year: int = 2015,
    num_additional_years: int = 5,
    *,
    current_year: int | None = None,
    min_train_points: int = 3,
    loess_frac: float = 0.6,
    loess_degree: int = 2,
    tail_damping_years: Any = _AUTO_TAIL_DAMPING,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Forecast SI and AMR counts for all pathogen-antibiotic pairs."""
    if current_year is None:
        current_year = int(pd.Timestamp.today().year)
    target_year = int(current_year) + int(num_additional_years)

    pair_index_df = (
        si_df[["pathogen", "antibiotic"]]
        .drop_duplicates()
        .sort_values(["pathogen", "antibiotic"])
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    for pair in pair_index_df.itertuples(index=False):
        pathogen = str(pair.pathogen)
        antibiotic = str(pair.antibiotic)
        try:
            result = fit_predict_si_loess(
                si_df=si_df,
                pathogen=pathogen,
                antibiotic=antibiotic,
                fit_start_year=fit_start_year,
                fit_end_year=fit_end_year,
                num_additional_years=max(num_additional_years, target_year - fit_end_year),
                min_train_points=min_train_points,
                loess_frac=loess_frac,
                loess_degree=loess_degree,
                tail_damping_years=tail_damping_years,
            )
            target_row = result["pred_df"].loc[result["pred_df"]["year"] == int(target_year)].iloc[0]
            pair_df = result["pair_df"]
            train_df = result["train_df"]
            rows.append(
                {
                    "pathogen": pathogen,
                    "antibiotic": antibiotic,
                    "fit_start_year": int(fit_start_year),
                    "fit_end_year": int(fit_end_year),
                    "current_year": int(current_year),
                    "forecast_year": int(target_year),
                    "num_additional_years": int(num_additional_years),
                    "loess_frac": float(result["loess_fit"]["frac"]),
                    "loess_degree": int(result["loess_fit"]["degree"]),
                    "tail_damping_years": result["loess_fit"]["tail_damping_years"],
                    "predicted_SI_mean": float(target_row["predicted_SI"]),
                    "predicted_SI_std": float(target_row["predicted_std"]),
                    "predicted_SI_lower_1std": float(target_row["lower_1std"]),
                    "predicted_SI_upper_1std": float(target_row["upper_1std"]),
                    "predicted_count_mean": float(target_row["predicted_count_loess"]),
                    "predicted_count_lower_1std": float(target_row["predicted_count_loess_lower_1std"]),
                    "predicted_count_upper_1std": float(target_row["predicted_count_loess_upper_1std"]),
                    "latest_available_year": int(result["latest_available_year"]),
                    "pair_first_year": int(pair_df["end_year"].min()),
                    "pair_last_year": int(pair_df["end_year"].max()),
                    "n_pair_intervals": int(pair_df["end_year"].nunique()),
                    "n_train_intervals": int(train_df["end_year"].nunique()),
                    "amr_pair_total_count": float(pair_df["count"].sum()),
                    "status": "ok",
                    "error": None,
                }
            )
        except Exception as exc:  # pragma: no cover - notebook-facing recovery path
            rows.append(
                {
                    "pathogen": pathogen,
                    "antibiotic": antibiotic,
                    "fit_start_year": int(fit_start_year),
                    "fit_end_year": int(fit_end_year),
                    "current_year": int(current_year),
                    "forecast_year": int(target_year),
                    "num_additional_years": int(num_additional_years),
                    "loess_frac": float(loess_frac),
                    "loess_degree": int(loess_degree),
                    "tail_damping_years": (
                        np.nan
                        if tail_damping_years in (_AUTO_TAIL_DAMPING, None)
                        else float(tail_damping_years)
                    ),
                    "predicted_SI_mean": np.nan,
                    "predicted_SI_std": np.nan,
                    "predicted_SI_lower_1std": np.nan,
                    "predicted_SI_upper_1std": np.nan,
                    "predicted_count_mean": np.nan,
                    "predicted_count_lower_1std": np.nan,
                    "predicted_count_upper_1std": np.nan,
                    "latest_available_year": np.nan,
                    "pair_first_year": np.nan,
                    "pair_last_year": np.nan,
                    "n_pair_intervals": np.nan,
                    "n_train_intervals": np.nan,
                    "amr_pair_total_count": np.nan,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    forecast_df = pd.DataFrame(rows).sort_values(
        ["status", "predicted_SI_mean", "amr_pair_total_count"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    if save_path is None:
        save_path = (
            f"si_pair_forecasts_fit_{int(fit_start_year)}_{int(fit_end_year)}"
            f"_target_{int(target_year)}.parquet"
        )
    forecast_df.to_parquet(save_path, index=False)
    return forecast_df


def plot_si_prediction(
    si_df: pd.DataFrame,
    pathogen: str,
    antibiotic: str,
    fit_start_year: int,
    fit_end_year: int,
    num_additional_years: int = 5,
    *,
    figsize: tuple[int, int] = (12, 8),
    loess_frac: float = 0.6,
    loess_degree: int = 2,
    tail_damping_years: Any = _AUTO_TAIL_DAMPING,
) -> dict[str, Any]:
    """Plot observed SI, smoothed SI, and smoothed AMR counts for one pair."""
    result = fit_predict_si_loess(
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

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax2 = ax.twinx()
    ax.set_facecolor("#fbfbf8")
    ax2.set_facecolor("none")
    ax.set_zorder(2)
    ax2.set_zorder(1)
    ax.patch.set_alpha(0.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax2.spines["top"].set_visible(False)

    count_plot_df = pair_df[["end_year", "count", "bg_pathogen_n", "bg_antibiotic_n"]].copy()
    count_plot_df["count_plot"] = np.clip(count_plot_df["count"].astype(float), 1.0, None)
    count_plot_df["bg_pathogen_plot"] = np.clip(count_plot_df["bg_pathogen_n"].astype(float), 1.0, None)
    count_plot_df["bg_antibiotic_plot"] = np.clip(count_plot_df["bg_antibiotic_n"].astype(float), 1.0, None)
    bar_width = 1.35
    ax2.bar(
        count_plot_df["end_year"],
        count_plot_df["bg_pathogen_plot"],
        width=bar_width,
        color="#8fbf88",
        alpha=0.26,
        edgecolor="none",
        label="Pathogen background count",
        zorder=0,
    )
    ax2.bar(
        count_plot_df["end_year"],
        count_plot_df["bg_antibiotic_plot"],
        width=bar_width,
        color="#b9a7d1",
        alpha=0.34,
        edgecolor="none",
        label="Antibiotic background count",
        zorder=1,
    )
    ax2.bar(
        count_plot_df["end_year"],
        count_plot_df["count_plot"],
        width=bar_width,
        color="#9ecae1",
        alpha=0.68,
        edgecolor="none",
        label="Per-interval AMR count",
        zorder=2,
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Publication counts (log scale)", color="#4e6475")
    ax2.tick_params(axis="y", labelcolor="#4e6475", colors="#4e6475")
    ax2.grid(False)

    observed_fit_df = pair_df.loc[pair_df["end_year"] <= int(fit_end_year)].copy()
    observed_future_df = pair_df.loc[pair_df["end_year"] > int(fit_end_year)].copy()
    if not observed_fit_df.empty and not observed_future_df.empty:
        observed_future_df = pd.concat([observed_fit_df.tail(1), observed_future_df], ignore_index=True)

    ax.plot(
        observed_fit_df["end_year"],
        observed_fit_df["SI"],
        color="#111111",
        marker="o",
        linewidth=3.0,
        markersize=8.5,
        markerfacecolor="#f8fbff",
        markeredgecolor="#111111",
        markeredgewidth=2.0,
        label="Observed SI (fit region)",
        zorder=3,
    )
    if not observed_future_df.empty:
        ax.plot(
            observed_future_df["end_year"],
            observed_future_df["SI"],
            color="#111111",
            marker="o",
            linewidth=3.0,
            linestyle="--",
            markersize=8.5,
            markerfacecolor="#f8fbff",
            markeredgecolor="#111111",
            markeredgewidth=2.0,
            label="Observed SI (after fit window)",
            zorder=3,
        )

    fit_pred_df = pred_df.loc[pred_df["year"] <= int(fit_end_year)].copy()
    forecast_pred_df = pred_df.loc[pred_df["year"] >= int(fit_end_year)].copy()

    x_min = min(pair_df["end_year"].min(), pred_df["year"].min()) - 1.2
    x_max = max(pair_df["end_year"].max(), pred_df["year"].max()) + 1.2
    ax.set_xlim(x_min, x_max)
    ax.axvline(fit_end_year, color="#6b7280", linestyle="--", linewidth=1.8, alpha=0.9, zorder=2)
    ax.text(
        fit_end_year + 0.2,
        0.035,
        f"fit end = {fit_end_year}",
        transform=ax.get_xaxis_transform(),
        color="#374151",
        fontsize=11,
        fontweight="semibold",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 2.5},
    )

    ax.plot(
        fit_pred_df["year"],
        fit_pred_df["predicted_SI"],
        color="#d62828",
        linewidth=3.0,
        label="LOESS SI (fit region)",
        zorder=5,
    )
    ax.plot(
        forecast_pred_df["year"],
        forecast_pred_df["predicted_SI"],
        color="#d62828",
        linewidth=3.0,
        linestyle="--",
        label="LOESS SI (forecast region)",
        zorder=5,
    )
    ax.fill_between(
        pred_df["year"],
        pred_df["lower_1std"],
        pred_df["upper_1std"],
        color="#f7b2b0",
        alpha=0.28,
        label="LOESS SI ±1 std",
        zorder=4,
    )

    ax2.plot(
        fit_pred_df["year"],
        np.clip(fit_pred_df["predicted_count_loess"], 1.0, None),
        color="#4e8fb3",
        linewidth=2.8,
        label="LOESS AMR count (fit region)",
        zorder=2,
    )
    ax2.plot(
        forecast_pred_df["year"],
        np.clip(forecast_pred_df["predicted_count_loess"], 1.0, None),
        color="#4e8fb3",
        linewidth=2.8,
        linestyle="--",
        label="LOESS AMR count (forecast region)",
        zorder=2,
    )
    ax2.fill_between(
        pred_df["year"],
        np.clip(pred_df["predicted_count_loess_lower_1std"], 1.0, None),
        np.clip(pred_df["predicted_count_loess_upper_1std"], 1.0, None),
        color="#bcd8e8",
        alpha=0.22,
        label="LOESS AMR count ±1 std",
        zorder=1,
    )

    ax.set_xlabel("Interval end year")
    ax.set_ylabel("SI = fitted log-count - pooled expected log-count")
    ax.set_title(
        f"Scientific Interest (SI) LOESS count fit: {pathogen} vs {antibiotic}",
        fontsize=16,
        pad=10,
    )
    ax.grid(True, axis="y", color="#c7c7c7", alpha=0.45, linewidth=0.8)
    ax.grid(False, axis="x")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handlelength=2.6,
        columnspacing=1.4,
    )
    plt.tight_layout(rect=(0, 0.11, 1, 1))
    plt.show()

    result["figure"] = fig
    return result


def plot_yearly_si_pair(
    si_df: pd.DataFrame,
    pathogen: str,
    antibiotic: str,
    fit_start_year: int,
    fit_end_year: int,
    num_additional_years: int = 5,
    *,
    figsize: tuple[int, int] = (12, 8),
    loess_frac: float = 0.6,
    loess_degree: int = 2,
    tail_damping_years: Any = _AUTO_TAIL_DAMPING,
) -> dict[str, Any]:
    """Plot observed yearly SI, smoothed yearly SI, and smoothed yearly AMR counts for one pair."""
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

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax2 = ax.twinx()
    ax.set_facecolor("#fbfbf8")
    ax2.set_facecolor("none")
    ax.set_zorder(2)
    ax2.set_zorder(1)
    ax.patch.set_alpha(0.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax2.spines["top"].set_visible(False)

    count_plot_df = pair_df[["year", "count", "bg_pathogen_n", "bg_antibiotic_n"]].copy()
    count_plot_df["count_plot"] = np.clip(count_plot_df["count"].astype(float), 1.0, None)
    count_plot_df["bg_pathogen_plot"] = np.clip(count_plot_df["bg_pathogen_n"].astype(float), 1.0, None)
    count_plot_df["bg_antibiotic_plot"] = np.clip(count_plot_df["bg_antibiotic_n"].astype(float), 1.0, None)
    bar_width = 0.85
    ax2.bar(
        count_plot_df["year"],
        count_plot_df["bg_pathogen_plot"],
        width=bar_width,
        color="#8fbf88",
        alpha=0.26,
        edgecolor="none",
        label="Pathogen background count",
        zorder=0,
    )
    ax2.bar(
        count_plot_df["year"],
        count_plot_df["bg_antibiotic_plot"],
        width=bar_width,
        color="#b9a7d1",
        alpha=0.34,
        edgecolor="none",
        label="Antibiotic background count",
        zorder=1,
    )
    ax2.bar(
        count_plot_df["year"],
        count_plot_df["count_plot"],
        width=bar_width,
        color="#9ecae1",
        alpha=0.68,
        edgecolor="none",
        label="Per-year AMR count",
        zorder=2,
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Publication counts (log scale)", color="#4e6475")
    ax2.tick_params(axis="y", labelcolor="#4e6475", colors="#4e6475")
    ax2.grid(False)

    observed_fit_df = pair_df.loc[pair_df["year"] <= int(fit_end_year)].copy()
    observed_future_df = pair_df.loc[pair_df["year"] > int(fit_end_year)].copy()
    if not observed_fit_df.empty and not observed_future_df.empty:
        observed_future_df = pd.concat([observed_fit_df.tail(1), observed_future_df], ignore_index=True)

    ax.plot(
        observed_fit_df["year"],
        observed_fit_df["SI"],
        color="#111111",
        marker="o",
        linewidth=3.0,
        markersize=6.5,
        markerfacecolor="#f8fbff",
        markeredgecolor="#111111",
        markeredgewidth=1.8,
        label="Observed SI (fit region)",
        zorder=3,
    )
    if not observed_future_df.empty:
        ax.plot(
            observed_future_df["year"],
            observed_future_df["SI"],
            color="#111111",
            marker="o",
            linewidth=3.0,
            linestyle="--",
            markersize=6.5,
            markerfacecolor="#f8fbff",
            markeredgecolor="#111111",
            markeredgewidth=1.8,
            label="Observed SI (after fit window)",
            zorder=3,
        )

    fit_pred_df = pred_df.loc[pred_df["year"] <= int(fit_end_year)].copy()
    forecast_pred_df = pred_df.loc[pred_df["year"] >= int(fit_end_year)].copy()

    x_min = min(pair_df["year"].min(), pred_df["year"].min()) - 0.8
    x_max = max(pair_df["year"].max(), pred_df["year"].max()) + 0.8
    ax.set_xlim(x_min, x_max)
    ax.axvline(fit_end_year, color="#6b7280", linestyle="--", linewidth=1.8, alpha=0.9, zorder=2)
    ax.text(
        fit_end_year + 0.2,
        0.035,
        f"fit end = {fit_end_year}",
        transform=ax.get_xaxis_transform(),
        color="#374151",
        fontsize=11,
        fontweight="semibold",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 2.5},
    )

    ax.plot(
        fit_pred_df["year"],
        fit_pred_df["predicted_SI"],
        color="#d62828",
        linewidth=3.0,
        label="LOESS SI (fit region)",
        zorder=5,
    )
    ax.plot(
        forecast_pred_df["year"],
        forecast_pred_df["predicted_SI"],
        color="#d62828",
        linewidth=3.0,
        linestyle="--",
        label="LOESS SI (forecast region)",
        zorder=5,
    )
    ax.fill_between(
        pred_df["year"],
        pred_df["lower_1std"],
        pred_df["upper_1std"],
        color="#f7b2b0",
        alpha=0.28,
        label="LOESS SI ±1 std",
        zorder=4,
    )

    ax2.plot(
        fit_pred_df["year"],
        np.clip(fit_pred_df["predicted_count_loess"], 1.0, None),
        color="#4e8fb3",
        linewidth=2.8,
        label="LOESS AMR count (fit region)",
        zorder=2,
    )
    ax2.plot(
        forecast_pred_df["year"],
        np.clip(forecast_pred_df["predicted_count_loess"], 1.0, None),
        color="#4e8fb3",
        linewidth=2.8,
        linestyle="--",
        label="LOESS AMR count (forecast region)",
        zorder=2,
    )
    ax2.fill_between(
        pred_df["year"],
        np.clip(pred_df["predicted_count_loess_lower_1std"], 1.0, None),
        np.clip(pred_df["predicted_count_loess_upper_1std"], 1.0, None),
        color="#bcd8e8",
        alpha=0.22,
        label="LOESS AMR count ±1 std",
        zorder=1,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("SI = fitted log-count - pooled expected log-count")
    ax.set_title(
        f"Yearly Scientific Interest (SI) LOESS count fit: {pathogen} vs {antibiotic}",
        fontsize=16,
        pad=10,
    )
    ax.grid(True, axis="y", color="#c7c7c7", alpha=0.45, linewidth=0.8)
    ax.grid(False, axis="x")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handlelength=2.6,
        columnspacing=1.4,
    )
    plt.tight_layout(rect=(0, 0.11, 1, 1))
    plt.show()

    result["figure"] = fig
    return result


def _slugify_plot_filename_component(value: str) -> str:
    """Return a filesystem-safe component for plot filenames."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return slug or "unknown"


def _create_plot_output_dir(base_dir: str = ".") -> str:
    """Create a new folder for a batch of Pareto SI plots."""
    root_name = "pareto_pair_si_plots"
    candidate = os.path.join(base_dir, root_name)
    suffix = 1
    while os.path.exists(candidate):
        suffix += 1
        candidate = os.path.join(base_dir, f"{root_name}_{suffix}")
    os.makedirs(candidate, exist_ok=False)
    return candidate


def display_si_amr_pareto_front(
    si_df: pd.DataFrame,
    *,
    n_pareto_shells: int = 1,
    si_col: str = "mean_SI",
    count_col: str = "mean_count",
) -> pd.DataFrame:
    """Compute and display Pareto shells maximizing average SI and average AMR count."""
    if n_pareto_shells <= 0:
        raise ValueError("n_pareto_shells must be >= 1.")

    required_cols = {"pathogen", "antibiotic", "SI", "count"}
    missing = required_cols - set(si_df.columns)
    if missing:
        raise KeyError(f"si_df is missing required columns: {sorted(missing)}")

    work = si_df.copy()
    work["SI"] = pd.to_numeric(work["SI"], errors="coerce")
    work["count"] = pd.to_numeric(work["count"], errors="coerce")
    agg_spec: dict[str, tuple[str, str]] = {
        "mean_SI": ("SI", "mean"),
        "mean_count": ("count", "mean"),
        "total_count": ("count", "sum"),
        "n_intervals": ("count", "size"),
    }
    if "end_year" in work.columns:
        agg_spec["latest_interval_end"] = ("end_year", "max")
    if "year" in work.columns:
        agg_spec["n_years"] = ("year", "nunique")
        agg_spec["latest_year"] = ("year", "max")

    work = (
        work.dropna(subset=["SI", "count"])
        .groupby(["pathogen", "antibiotic"], as_index=False)
        .agg(**agg_spec)
    )
    if work.empty:
        raise ValueError("No valid rows remain after filtering missing objective values.")

    def _pareto_mask(values: np.ndarray) -> np.ndarray:
        is_efficient = np.ones(values.shape[0], dtype=bool)
        for i, candidate in enumerate(values):
            if not is_efficient[i]:
                continue
            dominates = np.all(values >= candidate, axis=1) & np.any(values > candidate, axis=1)
            if dominates.any():
                is_efficient[i] = False
        return is_efficient

    remaining = work.copy()
    shell_frames = []
    for shell_idx in range(1, int(n_pareto_shells) + 1):
        if remaining.empty:
            break
        objective_values = remaining[[si_col, count_col]].to_numpy(dtype=float)
        shell_mask = _pareto_mask(objective_values)
        shell_df = remaining.loc[shell_mask].copy()
        shell_df["pareto_shell"] = shell_idx
        shell_df = shell_df.sort_values([si_col, count_col], ascending=[False, False]).reset_index(drop=True)
        shell_frames.append(shell_df)
        remaining = remaining.loc[~shell_mask].copy()

    if not shell_frames:
        raise ValueError("Unable to compute any Pareto shell from si_df.")

    pareto_df = pd.concat(shell_frames, ignore_index=True)
    display_cols = [
        "pareto_shell",
        "pathogen",
        "antibiotic",
        si_col,
        count_col,
    ]
    optional_cols = [
        "total_count",
        "n_intervals",
        "n_years",
        "latest_interval_end",
        "latest_year",
    ]
    display_cols.extend([col for col in optional_cols if col in pareto_df.columns])
    display(pareto_df[display_cols])
    return pareto_df


def plot_si_prediction_for_pareto_pairs(
    si_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    *,
    fit_start_year: int,
    fit_end_year: int,
    num_additional_years: int = 5,
    loess_frac: float = 0.7,
    loess_degree: int = 2,
    tail_damping_years: Any = _AUTO_TAIL_DAMPING,
    n_pareto_shells: int | None = None,
    save_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Plot SI prediction charts for selected pathogen-antibiotic Pareto pairs."""
    required_cols = {"pathogen", "antibiotic"}
    missing = required_cols - set(pareto_df.columns)
    if missing:
        raise KeyError(f"pareto_df is missing required columns: {sorted(missing)}")

    selected_pairs = pareto_df.copy()
    if n_pareto_shells is not None:
        if "pareto_shell" not in selected_pairs.columns:
            raise KeyError("pareto_df must include 'pareto_shell' when n_pareto_shells is provided.")
        selected_pairs = selected_pairs.loc[selected_pairs["pareto_shell"] <= int(n_pareto_shells)].copy()

    selected_pairs = selected_pairs[["pathogen", "antibiotic"]].drop_duplicates().reset_index(drop=True)
    output_dir = save_dir or _create_plot_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for pair in selected_pairs.itertuples(index=False):
        pathogen = str(pair.pathogen)
        antibiotic = str(pair.antibiotic)
        print(f"Plotting {pathogen} vs {antibiotic}")
        result = plot_si_prediction(
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
        pathogen_slug = _slugify_plot_filename_component(pathogen)
        antibiotic_slug = _slugify_plot_filename_component(antibiotic)
        png_path = os.path.join(output_dir, f"{pathogen_slug}__{antibiotic_slug}.png")
        result["figure"].savefig(png_path, dpi=200, bbox_inches="tight")
        result["png_path"] = png_path
        results.append(result)
    print(f"Saved {len(results)} plot PNGs to {output_dir}")
    return results


def plot_yearly_si_prediction_for_pareto_pairs(
    si_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    *,
    fit_start_year: int,
    fit_end_year: int,
    num_additional_years: int = 5,
    loess_frac: float = 0.7,
    loess_degree: int = 2,
    tail_damping_years: Any = _AUTO_TAIL_DAMPING,
    n_pareto_shells: int | None = None,
    save_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Plot yearly SI prediction charts for selected pathogen-antibiotic Pareto pairs."""
    required_cols = {"pathogen", "antibiotic"}
    missing = required_cols - set(pareto_df.columns)
    if missing:
        raise KeyError(f"pareto_df is missing required columns: {sorted(missing)}")

    selected_pairs = pareto_df.copy()
    if n_pareto_shells is not None:
        if "pareto_shell" not in selected_pairs.columns:
            raise KeyError("pareto_df must include 'pareto_shell' when n_pareto_shells is provided.")
        selected_pairs = selected_pairs.loc[selected_pairs["pareto_shell"] <= int(n_pareto_shells)].copy()

    selected_pairs = selected_pairs[["pathogen", "antibiotic"]].drop_duplicates().reset_index(drop=True)
    output_dir = save_dir or _create_plot_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for pair in selected_pairs.itertuples(index=False):
        pathogen = str(pair.pathogen)
        antibiotic = str(pair.antibiotic)
        print(f"Plotting {pathogen} vs {antibiotic}")
        try:
            result = plot_yearly_si_pair(
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
        except Exception as exc:  # pragma: no cover - notebook-facing recovery path
            print(f"Skipping {pathogen} vs {antibiotic}: {exc}")
            continue
        pathogen_slug = _slugify_plot_filename_component(pathogen)
        antibiotic_slug = _slugify_plot_filename_component(antibiotic)
        png_path = os.path.join(output_dir, f"{pathogen_slug}__{antibiotic_slug}.png")
        result["figure"].savefig(png_path, dpi=200, bbox_inches="tight")
        result["png_path"] = png_path
        results.append(result)
    print(f"Saved {len(results)} plot PNGs to {output_dir}")
    return results


# Backward-compatible aliases for earlier notebook cells / exports.
get_pair_si_series = build_pair_si_interval_series
_interpolate_expected_log_count_baseline = _interpolate_expected_log_count_curve
_fit_loess_log_count = _fit_loess_log_count_curve
_evaluate_loess_with_linear_tails = _predict_loess_log_count_with_linear_tails
fit_predict_si_gp = fit_predict_si_loess
forecast_si_for_all_pairs = forecast_all_pair_si_loess


__all__ = [
    "load_amr_background_df",
    "fit_global_pooled_si_model",
    "build_pair_si_interval_series",
    "build_pair_si_yearly_series",
    "fit_predict_si_loess",
    "fit_predict_yearly_si_loess",
    "forecast_all_pair_si_loess",
    "plot_si_prediction",
    "plot_yearly_si_pair",
    "display_si_amr_pareto_front",
    "plot_si_prediction_for_pareto_pairs",
    "plot_yearly_si_prediction_for_pareto_pairs",
    "get_pair_si_series",
    "fit_predict_si_gp",
    "forecast_si_for_all_pairs",
]
