"""Scientific-interest utilities for AMR interval analyses."""

from __future__ import annotations

import os
import re
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from skmisc.loess import loess
from sklearn.linear_model import Ridge

from .normalisation import BackgroundNormaliseConfig, normalise_by_background

_AUTO_TAIL_DAMPING = object()
_DEFAULT_RUNTIME_LOG_PATH = "scientific_interest.log"


def _log_transform(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Base-10 log transform used throughout the scientific-interest workflow."""
    return np.log10(1.0 + values)


def _inverse_log_transform(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Inverse of the base-10 log transform."""
    return np.power(10.0, values) - 1.0


def _get_runtime_logger(log_path: str = _DEFAULT_RUNTIME_LOG_PATH) -> logging.Logger:
    logger_name = f"{__name__}:{os.path.abspath(log_path)}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _log_runtime_warning(message: str, *, log_path: str = _DEFAULT_RUNTIME_LOG_PATH) -> None:
    _get_runtime_logger(log_path).warning(message)


def _configure_plotly_inline_renderer(pio: Any) -> None:
    """Prefer an inline Plotly renderer when running from notebooks."""
    current = (getattr(pio.renderers, "default", "") or "").strip()
    if current and current != "browser":
        return
    pio.renderers.default = "plotly_mimetype+notebook_connected"


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
    amr_background_df["log_count"] = _log_transform(pd.to_numeric(amr_background_df["count"], errors="coerce"))
    amr_background_df["log_bg_pathogen_n"] = _log_transform(pd.to_numeric(amr_background_df["bg_pathogen_n"], errors="coerce"))
    amr_background_df["log_bg_antibiotic_n"] = _log_transform(pd.to_numeric(amr_background_df["bg_antibiotic_n"], errors="coerce"))
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
    work["expected_count_global"] = np.clip(_inverse_log_transform(work["expected_log_count_global"]), 0.0, None)
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
    pair_df["log_count"] = _log_transform(pair_df["count"].astype(float))
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
    pair_df["log_count"] = _log_transform(pair_df["count"].astype(float))
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
    log_path: str = _DEFAULT_RUNTIME_LOG_PATH,
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
        _log_runtime_warning(
            f"Quadratic LOESS was numerically unstable ({exc}). Falling back to linear LOESS.",
            log_path=log_path,
        )
        try:
            fitted_log_counts = _loess_predict(years, log_counts, years, frac=loess_frac, degree=1)
            fitted_degree = 1
            fit_method = "loess_degree_1"
            linear_slope = None
            linear_intercept = None
            poly_coeffs = None
        except ValueError as fallback_exc:
            _log_runtime_warning(
                f"Linear LOESS was numerically unstable ({fallback_exc}). Falling back to quadratic polynomial fit.",
                log_path=log_path,
            )
            try:
                poly_coeffs = np.polyfit(years, log_counts, deg=2)
                fitted_log_counts = np.polyval(poly_coeffs, years)
                fitted_degree = 2
                fit_method = "quadratic_regression"
                linear_slope = None
                linear_intercept = None
            except np.linalg.LinAlgError as poly_exc:
                _log_runtime_warning(
                    f"Quadratic polynomial fit was numerically unstable ({poly_exc}). Falling back to ordinary linear regression.",
                    log_path=log_path,
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
    detailed_comparison_df = comparison_df.copy()
    compact_predicted_count = np.rint(comparison_df["predicted_count_loess"]).astype(int)
    compact_observed_count = np.rint(comparison_df["observed_count"]).astype(int)
    compact_diff = compact_observed_count - compact_predicted_count
    compact_pct_error = np.where(
        compact_observed_count != 0,
        100.0 * compact_diff / compact_observed_count,
        np.nan,
    )
    comparison_df = comparison_df[
        ["year"]
    ].copy()
    comparison_df["observed_count"] = compact_observed_count
    comparison_df["predicted_count"] = compact_predicted_count
    comparison_df["diff"] = compact_diff
    comparison_df["pct_error"] = compact_pct_error

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
        "detailed_comparison_df": detailed_comparison_df,
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


def rank_yearly_pairs_by_future_si_shift(
    si_df: pd.DataFrame,
    *,
    fit_start_year: int = 1970,
    fit_end_year: int | None = None,
    num_future_years: int = 20,
    reference_year: int | None = None,
    min_train_points: int = 3,
    loess_frac: float = 0.6,
    loess_degree: int = 2,
    tail_damping_years: Any = _AUTO_TAIL_DAMPING,
    save_path: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Rank yearly pathogen-antibiotic pairs by current vs future predicted SI.

    The current rank is defined at ``reference_year`` and the future rank at
    ``latest_available_year + num_future_years`` for each pair. Both values come
    from the yearly LOESS count model with SI derived from the predicted
    log-count trajectory.

    A positive ``rank_change`` means the pair moves up in the future ranking.
    """
    def _safe_pct_error(error: pd.Series, observed: pd.Series) -> pd.Series:
        observed = pd.to_numeric(observed, errors="coerce")
        error = pd.to_numeric(error, errors="coerce")
        denom = observed.abs()
        return pd.Series(
            np.where(denom > 0, 100.0 * error / denom, np.nan),
            index=error.index,
            dtype=float,
        )

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
    year_num = pd.to_numeric(si_df["year"], errors="coerce")
    train_window_mask = year_num.between(int(fit_start_year), int(fit_end_year))
    pair_stats_df = (
        si_df.loc[:, ["pathogen", "antibiotic"]].copy()
        .assign(year_num=year_num, in_train_window=train_window_mask)
        .groupby(["pathogen", "antibiotic"], as_index=False)
        .agg(
            n_pair_years=("year_num", lambda s: int(pd.Series(s).dropna().nunique())),
            n_train_years=(
                "year_num",
                lambda s: int(pd.Series(s[train_window_mask.loc[s.index]]).dropna().nunique()),
            ),
            latest_available_year=("year_num", lambda s: float(pd.Series(s).dropna().max()) if pd.Series(s).notna().any() else np.nan),
        )
        .sort_values(["pathogen", "antibiotic"])
        .reset_index(drop=True)
    )
    eligible_pair_df = pair_stats_df.loc[pair_stats_df["n_train_years"] >= int(min_train_points)].copy()
    prefiltered_pair_df = pair_stats_df.loc[pair_stats_df["n_train_years"] < int(min_train_points)].copy()
    pair_index_df = eligible_pair_df[["pathogen", "antibiotic"]].reset_index(drop=True)
    total_pairs = int(len(pair_index_df))

    rows: list[dict[str, Any]] = []
    if not prefiltered_pair_df.empty:
        if verbose:
            print(
                "Prefilter removed "
                f"{len(prefiltered_pair_df)} pairs with fewer than {min_train_points} observed years "
                f"between {fit_start_year} and {fit_end_year}.",
                flush=True,
            )
        prefilter_error = (
            f"Need at least {min_train_points} observed years between {fit_start_year} and {fit_end_year}."
        )
        for pair in prefiltered_pair_df.itertuples(index=False):
            pair_future_year = (
                int(pair.latest_available_year) + int(num_future_years)
                if pd.notna(pair.latest_available_year)
                else np.nan
            )
            rows.append(
                {
                    "pathogen": str(pair.pathogen),
                    "antibiotic": str(pair.antibiotic),
                    "reference_year": reference_year,
                    "future_year": pair_future_year,
                    "num_future_years": int(num_future_years),
                    "fit_start_year": fit_start_year,
                    "fit_end_year": fit_end_year,
                    "latest_available_year": (
                        int(pair.latest_available_year) if pd.notna(pair.latest_available_year) else np.nan
                    ),
                    "n_pair_years": int(pair.n_pair_years),
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
                    "tail_damping_years": (
                        np.nan
                        if tail_damping_years in (_AUTO_TAIL_DAMPING, None)
                        else float(tail_damping_years)
                    ),
                    "status": "failed",
                    "error": f"{prefilter_error} Found {int(pair.n_train_years)}.",
                }
            )

    for idx, pair in enumerate(pair_index_df.itertuples(index=False), start=1):
        pathogen = str(pair.pathogen)
        antibiotic = str(pair.antibiotic)
        if verbose:
            remaining = total_pairs - idx
            print(f"[{idx}/{total_pairs}] ranking {pathogen} vs {antibiotic} ({remaining} remaining)", flush=True)
        try:
            pair_df = build_pair_si_yearly_series(si_df, pathogen, antibiotic)
            latest_available_year = int(pair_df["year"].max())
            future_year = int(latest_available_year + int(num_future_years))
            horizon_years = int(num_future_years)
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
            )

            pred_df = result["pred_df"]
            current_row = pred_df.loc[pred_df["year"] == reference_year]
            future_row = pred_df.loc[pred_df["year"] == future_year]
            if current_row.empty:
                raise ValueError(f"Missing current-year prediction for reference_year={reference_year}.")
            if future_row.empty:
                raise ValueError(f"Missing future-year prediction for future_year={future_year}.")

            current_row = current_row.iloc[0]
            future_row = future_row.iloc[0]
            observed_current_df = pair_df.loc[pair_df["year"] == reference_year]
            current_source = "observed" if not observed_current_df.empty else "predicted"
            current_count = (
                float(observed_current_df["count"].iloc[0])
                if not observed_current_df.empty and "count" in observed_current_df.columns
                else np.nan
            )
            latest_available_df = pair_df.loc[pair_df["year"] == latest_available_year]
            latest_available_count = (
                float(latest_available_df["count"].iloc[0])
                if not latest_available_df.empty and "count" in latest_available_df.columns
                else np.nan
            )
            latest_available_si = (
                float(latest_available_df["SI"].iloc[0])
                if not latest_available_df.empty and "SI" in latest_available_df.columns
                else np.nan
            )
            latest_available_pred_df = pred_df.loc[pred_df["year"] == latest_available_year]
            latest_available_predicted_count = (
                float(latest_available_pred_df["predicted_count_loess"].iloc[0])
                if not latest_available_pred_df.empty
                else np.nan
            )
            latest_available_predicted_si = (
                float(latest_available_pred_df["predicted_SI"].iloc[0])
                if not latest_available_pred_df.empty
                else np.nan
            )
            holdout_df = result["detailed_comparison_df"].loc[
                result["detailed_comparison_df"]["year"] > fit_end_year
            ].copy()
            holdout_df["si_pct_error"] = _safe_pct_error(
                holdout_df["prediction_error"],
                holdout_df["observed_SI"],
            )
            count_error_df = result["detailed_comparison_df"].loc[
                result["detailed_comparison_df"]["year"].between(fit_end_year, latest_available_year)
            ].copy()
            count_error_df["count_pct_error_loess"] = _safe_pct_error(
                count_error_df["count_prediction_error_loess"],
                count_error_df["observed_count"],
            )
            count_error_df["count_abs_pct_error_loess"] = count_error_df["count_pct_error_loess"].abs()
            n_count_error_years = int(len(count_error_df))
            latest_count_error_row = count_error_df.sort_values("year").tail(1)
            if latest_count_error_row.empty:
                latest_count_pct_error = np.nan
                latest_count_abs_pct_error = np.nan
                post_fit_count_pct_error_mean = np.nan
                post_fit_count_abs_pct_error_mean = np.nan
            else:
                latest_count_error = latest_count_error_row.iloc[0]
                latest_count_pct_error = float(latest_count_error["count_pct_error_loess"])
                latest_count_abs_pct_error = float(latest_count_error["count_abs_pct_error_loess"])
                post_fit_count_pct_error_mean = float(count_error_df["count_pct_error_loess"].mean())
                post_fit_count_abs_pct_error_mean = float(count_error_df["count_abs_pct_error_loess"].mean())
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
                latest_holdout = latest_holdout_row.iloc[0]
                latest_holdout_year = int(latest_holdout["year"])
                latest_observed_si = float(latest_holdout["observed_SI"])
                latest_predicted_si = float(latest_holdout["predicted_SI"])
                latest_si_error = float(latest_holdout["prediction_error"])
                latest_si_abs_error = float(latest_holdout["abs_error"])
                latest_si_pct_error = float(latest_holdout["si_pct_error"])
                post_fit_si_error_mean = float(holdout_df["prediction_error"].mean())
                post_fit_si_abs_error_mean = float(holdout_df["abs_error"].mean())
                post_fit_si_pct_error_mean = float(holdout_df["si_pct_error"].mean())
                post_fit_si_abs_pct_error_mean = float(holdout_df["si_pct_error"].abs().mean())

            rows.append(
                {
                    "pathogen": pathogen,
                    "antibiotic": antibiotic,
                    "reference_year": reference_year,
                    "future_year": future_year,
                    "num_future_years": int(num_future_years),
                    "fit_start_year": fit_start_year,
                    "fit_end_year": fit_end_year,
                    "latest_available_year": latest_available_year,
                    "n_pair_years": int(pair_df["year"].nunique()),
                    "current_si_source": current_source,
                    "current_count": current_count,
                    "latest_available_count": latest_available_count,
                    "latest_available_SI": latest_available_si,
                    "latest_available_predicted_count": latest_available_predicted_count,
                    "latest_available_predicted_SI": latest_available_predicted_si,
                    "observed_current_SI": (
                        float(observed_current_df["SI"].iloc[0]) if not observed_current_df.empty else np.nan
                    ),
                    "current_predicted_SI": float(current_row["predicted_SI"]),
                    "current_predicted_count": float(current_row["predicted_count_loess"]),
                    "future_predicted_SI": float(future_row["predicted_SI"]),
                    "future_predicted_count": float(future_row["predicted_count_loess"]),
                    "si_change": float(future_row["predicted_SI"] - current_row["predicted_SI"]),
                    "future_minus_latest_SI": float(future_row["predicted_SI"] - latest_available_si),
                    "count_change": float(future_row["predicted_count_loess"] - current_row["predicted_count_loess"]),
                    "n_count_error_years": n_count_error_years,
                    "latest_count_pct_error": latest_count_pct_error,
                    "latest_count_abs_pct_error": latest_count_abs_pct_error,
                    "post_fit_count_pct_error_mean": post_fit_count_pct_error_mean,
                    "post_fit_count_abs_pct_error_mean": post_fit_count_abs_pct_error_mean,
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
        except Exception as exc:  # pragma: no cover - notebook-facing recovery path
            if verbose:
                print(f"[{idx}/{total_pairs}] skipping {pathogen} vs {antibiotic}: {exc}", flush=True)
            rows.append(
                {
                    "pathogen": pathogen,
                    "antibiotic": antibiotic,
                    "reference_year": reference_year,
                    "future_year": np.nan,
                    "num_future_years": int(num_future_years),
                    "fit_start_year": fit_start_year,
                    "fit_end_year": fit_end_year,
                    "latest_available_year": np.nan,
                    "n_pair_years": np.nan,
                    "current_si_source": None,
                    "current_count": np.nan,
                    "latest_available_count": np.nan,
                    "latest_available_SI": np.nan,
                    "latest_available_predicted_count": np.nan,
                    "latest_available_predicted_SI": np.nan,
                    "observed_current_SI": np.nan,
                    "current_predicted_SI": np.nan,
                    "current_predicted_count": np.nan,
                    "future_predicted_SI": np.nan,
                    "future_predicted_count": np.nan,
                    "si_change": np.nan,
                    "future_minus_latest_SI": np.nan,
                    "count_change": np.nan,
                    "n_count_error_years": np.nan,
                    "latest_count_pct_error": np.nan,
                    "latest_count_abs_pct_error": np.nan,
                    "post_fit_count_pct_error_mean": np.nan,
                    "post_fit_count_abs_pct_error_mean": np.nan,
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
                    "tail_damping_years": (
                        np.nan
                        if tail_damping_years in (_AUTO_TAIL_DAMPING, None)
                        else float(tail_damping_years)
                    ),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    ranking_df = pd.DataFrame(rows)
    ok_mask = ranking_df["status"].eq("ok")
    ranking_df.loc[ok_mask, "current_rank"] = (
        ranking_df.loc[ok_mask, "current_predicted_SI"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    ranking_df.loc[ok_mask, "future_rank"] = (
        ranking_df.loc[ok_mask, "future_predicted_SI"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    ranking_df["rank_change"] = ranking_df["current_rank"] - ranking_df["future_rank"]
    ranking_df["abs_rank_change"] = ranking_df["rank_change"].abs()

    preferred_cols = [
        "pathogen",
        "antibiotic",
        "reference_year",
        "current_rank",
        "current_predicted_SI",
        "future_year",
        "future_rank",
        "future_predicted_SI",
        "rank_change",
        "abs_rank_change",
    ]
    remaining_cols = [col for col in ranking_df.columns if col not in preferred_cols]
    ranking_df = ranking_df[preferred_cols + remaining_cols]
    ranking_df = ranking_df.sort_values(
        ["status", "abs_rank_change", "future_predicted_SI", "current_predicted_SI"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    if save_path is None:
        save_path = (
            f"si_yearly_rank_shift_fit_{fit_start_year}_{fit_end_year}"
            f"_ref_{reference_year}_future_plus_{int(num_future_years)}y.parquet"
        )
    ranking_df.to_parquet(save_path, index=False)
    if verbose:
        n_ok = int(ranking_df["status"].eq("ok").sum())
        n_failed = int(ranking_df["status"].eq("failed").sum())
        print(
            f"Saved yearly SI rank shifts for {n_ok} pairs"
            f"{'' if n_failed == 0 else f' ({n_failed} failed)'} to {save_path}"
            ,
            flush=True,
        )
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
    """Filter and rank interesting yearly SI cases from ``rank_df``.

    The default policy keeps pairs that:
    - have non-trivial current and future predicted AMR volume
    - are predicted to gain SI meaningfully
    - have acceptable post-fit SI holdout error on observed years after ``fit_end_year``

    The returned frame includes:
    - boolean pass/fail flags for each filter component
    - ``reliability_penalty`` based on holdout SI error
    - ``interest_score`` that rewards SI gain and predicted volume, while penalizing unreliability

    If the strict filter yields no rows, the function falls back to a softer
    score-ranked shortlist so that notebooks always return something usable.
    """
    required_cols = {
        "status",
        "current_predicted_count",
        "future_predicted_count",
        "si_change",
    }
    missing = required_cols - set(rank_df.columns)
    if missing:
        raise KeyError(f"rank_df is missing required columns: {sorted(missing)}")

    work = rank_df.copy()
    work = work.loc[work["status"].eq("ok")].copy()

    # Backward compatibility: older rank parquet files may not include
    # post-fit validation metrics. Fill them with NaN so callers can still
    # use the selector with require_post_fit_validation=False.
    for col in [
        "post_fit_si_abs_error_mean",
        "post_fit_si_abs_pct_error_mean",
        "n_post_fit_observed_years",
    ]:
        if col not in work.columns:
            work[col] = np.nan

    work["passes_current_count"] = pd.to_numeric(
        work["current_predicted_count"], errors="coerce"
    ) >= float(min_current_predicted_count)
    work["passes_future_count"] = pd.to_numeric(
        work["future_predicted_count"], errors="coerce"
    ) >= float(min_future_predicted_count)
    current_count = pd.to_numeric(work["current_predicted_count"], errors="coerce")
    future_count = pd.to_numeric(work["future_predicted_count"], errors="coerce")
    future_to_current_ratio = future_count / current_count.where(current_count > 0)
    work["future_to_current_ratio"] = future_to_current_ratio
    work["passes_si_change"] = pd.to_numeric(
        work["si_change"], errors="coerce"
    ) >= float(min_si_change)
    if max_future_predicted_count is None:
        work["passes_future_count_ceiling"] = True
    else:
        work["passes_future_count_ceiling"] = future_count <= float(max_future_predicted_count)
    if max_future_to_current_ratio is None:
        work["passes_future_growth_ratio"] = True
    else:
        work["passes_future_growth_ratio"] = future_to_current_ratio <= float(max_future_to_current_ratio)
    post_fit_years = pd.to_numeric(work["n_post_fit_observed_years"], errors="coerce")
    post_fit_abs_error = pd.to_numeric(work["post_fit_si_abs_error_mean"], errors="coerce")
    post_fit_abs_pct_error = pd.to_numeric(work["post_fit_si_abs_pct_error_mean"], errors="coerce")
    has_post_fit_validation = (
        post_fit_years.notna()
        & post_fit_abs_error.notna()
        & post_fit_abs_pct_error.notna()
    )
    work["has_post_fit_validation"] = has_post_fit_validation
    work["passes_post_fit_years"] = post_fit_years >= int(min_post_fit_observed_years)
    work["passes_post_fit_abs_error"] = post_fit_abs_error <= float(max_post_fit_si_abs_error_mean)
    work["passes_post_fit_abs_pct_error"] = post_fit_abs_pct_error <= float(max_post_fit_si_abs_pct_error_mean)

    if require_post_fit_validation:
        work["passes_validation"] = (
            (~work["has_post_fit_validation"])
            | (
                work["passes_post_fit_years"]
                & work["passes_post_fit_abs_error"]
                & work["passes_post_fit_abs_pct_error"]
            )
        )
    else:
        work["passes_validation"] = (
            (~work["has_post_fit_validation"])
            | (
                work["passes_post_fit_abs_error"]
                & work["passes_post_fit_abs_pct_error"]
            )
        )

    work["is_interesting_case"] = (
        work["passes_current_count"]
        & work["passes_future_count"]
        & work["passes_future_count_ceiling"]
        & work["passes_future_growth_ratio"]
        & work["passes_si_change"]
        & work["passes_validation"]
    )

    abs_error = pd.to_numeric(work["post_fit_si_abs_error_mean"], errors="coerce")
    abs_pct_error = pd.to_numeric(work["post_fit_si_abs_pct_error_mean"], errors="coerce")
    missing_penalty = (
        abs_error.dropna().max() if abs_error.notna().any() else float(max_post_fit_si_abs_error_mean)
    ) + 0.01 * (
        abs_pct_error.dropna().max()
        if abs_pct_error.notna().any()
        else float(max_post_fit_si_abs_pct_error_mean)
    )
    work["reliability_penalty"] = abs_error.fillna(missing_penalty) + 0.01 * abs_pct_error.fillna(missing_penalty)
    future_count_for_score = pd.to_numeric(work["future_predicted_count"], errors="coerce").clip(
        lower=0,
        upper=float(max_count_for_scoring),
    )
    work["future_count_for_scoring"] = future_count_for_score
    work["interest_score"] = (
        pd.to_numeric(work["si_change"], errors="coerce")
        * _log_transform(future_count_for_score)
        / (1.0 + work["reliability_penalty"])
    )
    work.loc[~np.isfinite(work["interest_score"]), "interest_score"] = np.nan

    interesting_df = work.loc[work["is_interesting_case"]].copy()
    if interesting_df.empty:
        fallback_df = work.loc[
            work["passes_current_count"]
            & work["passes_future_count"]
            & work["passes_future_count_ceiling"]
            & work["passes_future_growth_ratio"]
            & (pd.to_numeric(work["si_change"], errors="coerce") > 0)
        ].copy()
        if fallback_df.empty:
            fallback_df = work.copy()
        interesting_df = fallback_df.sort_values(
            ["interest_score", "si_change", "future_predicted_SI"],
            ascending=[False, False, False],
            na_position="last",
        ).head(int(max(fallback_top_n, 1))).copy()
        interesting_df["is_interesting_case"] = True
        interesting_df["selection_mode"] = "fallback_top_n"
    else:
        interesting_df["selection_mode"] = "strict_filter"

    if sort_by not in interesting_df.columns:
        raise KeyError(f"sort_by={sort_by!r} is not a column in the filtered result.")

    preferred_cols = [
        "pathogen",
        "antibiotic",
        "selection_mode",
        "interest_score",
        "si_change",
        "current_predicted_count",
        "future_predicted_count",
        "future_count_for_scoring",
        "future_to_current_ratio",
        "current_rank",
        "future_rank",
        "rank_change",
        "n_post_fit_observed_years",
        "post_fit_si_abs_error_mean",
        "post_fit_si_abs_pct_error_mean",
        "reliability_penalty",
    ]
    remaining_cols = [col for col in interesting_df.columns if col not in preferred_cols]
    interesting_df = interesting_df[preferred_cols + remaining_cols]
    interesting_df = interesting_df.sort_values(
        [sort_by, "si_change", "future_predicted_SI"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return interesting_df


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
    verbose: bool = False,
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
        if verbose:
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
    if verbose:
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
    verbose: bool = False,
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
        if verbose:
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
            if verbose:
                print(f"Skipping {pathogen} vs {antibiotic}: {exc}")
            continue
        pathogen_slug = _slugify_plot_filename_component(pathogen)
        antibiotic_slug = _slugify_plot_filename_component(antibiotic)
        png_path = os.path.join(output_dir, f"{pathogen_slug}__{antibiotic_slug}.png")
        result["figure"].savefig(png_path, dpi=200, bbox_inches="tight")
        result["png_path"] = png_path
        results.append(result)
    if verbose:
        print(f"Saved {len(results)} plot PNGs to {output_dir}")
    return results


def plot_interactive_yearly_rank_shift(
    rank_df: pd.DataFrame,
    *,
    show: bool = True,
    display_name: bool = False,
    title: str = "Current to future predicted AMR count vs scientific interest",
    width: int = 1400,
    height: int = 950,
    current_color: str = "#1f77b4",
    future_color: str = "#ff7f0e",
    arrow_color_up: str = "rgba(31, 119, 180, 0.8)",
    arrow_color_down: str = "rgba(214, 39, 40, 0.8)",
):
    """Plot an interactive current-to-future AMR-count vs SI arrow chart for ranked pairs."""
    def _signed_log10(values: pd.Series | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.sign(arr) * np.log10(1.0 + np.abs(arr))

    required_cols = {
        "pathogen",
        "antibiotic",
        "current_predicted_count",
        "future_predicted_count",
        "current_predicted_SI",
        "future_predicted_SI",
    }
    missing = required_cols - set(rank_df.columns)
    if missing:
        raise KeyError(f"rank_df is missing required columns: {sorted(missing)}")

    try:
        import plotly.io as pio
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional notebook dependency
        raise ImportError("plotly is required for interactive SI plots.") from exc

    selected_cols = [
        "pathogen",
        "antibiotic",
        "current_predicted_count",
        "future_predicted_count",
        "current_predicted_SI",
        "future_predicted_SI",
    ]
    optional_plot_cols = [
        col for col in ["plot_current_predicted_count", "plot_future_predicted_count"] if col in rank_df.columns
    ]
    plot_df = rank_df[selected_cols + optional_plot_cols].copy()
    plot_df = plot_df.dropna(
        subset=[
            "current_predicted_count",
            "future_predicted_count",
            "current_predicted_SI",
            "future_predicted_SI",
        ]
    ).reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("No rows remain after dropping missing current/future prediction values.")

    if "plot_current_predicted_count" not in plot_df.columns:
        plot_df["plot_current_predicted_count"] = plot_df["current_predicted_count"]
    if "plot_future_predicted_count" not in plot_df.columns:
        plot_df["plot_future_predicted_count"] = plot_df["future_predicted_count"]

    plot_df = plot_df.loc[
        (pd.to_numeric(plot_df["plot_current_predicted_count"], errors="coerce") > 0)
        & (pd.to_numeric(plot_df["plot_future_predicted_count"], errors="coerce") > 0)
    ].reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("No rows remain with strictly positive current/future predicted counts for log-x plotting.")

    plot_df["label"] = plot_df.apply(
        lambda row: f"{str(row['pathogen']).strip()[:3]}<br>{str(row['antibiotic']).strip()[:3]}",
        axis=1,
    )
    plot_df["current_predicted_SI_signed_log10"] = _signed_log10(plot_df["current_predicted_SI"])
    plot_df["future_predicted_SI_signed_log10"] = _signed_log10(plot_df["future_predicted_SI"])
    si_delta = plot_df["future_predicted_SI"] - plot_df["current_predicted_SI"]
    si_scale = plot_df["current_predicted_SI"].abs().clip(lower=1e-6)
    plot_df["relative_si_change"] = (si_delta / si_scale).abs()
    max_relative_si_change = float(plot_df["relative_si_change"].max()) if not plot_df.empty else 0.0
    plot_df["edge_change_group"] = np.where(
        plot_df["future_predicted_SI"] > plot_df["current_predicted_SI"],
        "Up",
        "Down",
    )

    fig = go.Figure()

    edge_styles = [
        ("Up", arrow_color_up),
        ("Down", arrow_color_down),
    ]
    for group_name, group_color in edge_styles:
        group_df = plot_df.loc[plot_df["edge_change_group"] == group_name]
        if group_df.empty:
            continue
        for idx, row in enumerate(group_df.itertuples(index=False)):
            if max_relative_si_change > 0:
                line_width = 1.0 + 5.0 * (float(row.relative_si_change) / max_relative_si_change)
            else:
                line_width = 1.5
            fig.add_trace(
                go.Scatter(
                    x=[float(row.plot_current_predicted_count), float(row.plot_future_predicted_count)],
                    y=[
                        float(row.current_predicted_SI_signed_log10),
                        float(row.future_predicted_SI_signed_log10),
                    ],
                    mode="lines",
                    line=dict(color=group_color, width=line_width),
                    hoverinfo="skip",
                    name=group_name,
                    showlegend=idx == 0,
                )
            )

    fig.add_trace(
        go.Scatter(
            x=plot_df["plot_current_predicted_count"],
            y=plot_df["current_predicted_SI_signed_log10"],
            mode="markers+text" if display_name else "markers",
            text=plot_df["label"] if display_name else None,
            textposition="middle center" if display_name else None,
            marker=dict(color=current_color, size=8),
            customdata=plot_df[
                [
                    "pathogen",
                    "antibiotic",
                    "current_predicted_count",
                    "current_predicted_SI",
                    "future_predicted_count",
                    "future_predicted_SI",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Antibiotic: %{customdata[1]}<br>"
                "nAMR: %{customdata[2]:.0f} -> %{customdata[4]:.0f}<br>"
                "SI: %{customdata[3]:.3f} -> %{customdata[5]:.3f}<extra></extra>"
            ),
            name="Current",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["plot_future_predicted_count"],
            y=plot_df["future_predicted_SI_signed_log10"],
            mode="markers",
            marker=dict(color=future_color, size=9, symbol="diamond"),
            customdata=plot_df[
                [
                    "pathogen",
                    "antibiotic",
                    "current_predicted_count",
                    "current_predicted_SI",
                    "future_predicted_count",
                    "future_predicted_SI",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Antibiotic: %{customdata[1]}<br>"
                "nAMR: %{customdata[2]:.0f} -> %{customdata[4]:.0f}<br>"
                "SI: %{customdata[3]:.3f} -> %{customdata[5]:.3f}<extra></extra>"
            ),
            name="Future",
        )
    )

    si_tick_vals = np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0], dtype=float)
    si_tick_text = [f"{value:g}" for value in si_tick_vals]
    si_tick_positions = _signed_log10(si_tick_vals)

    fig.update_layout(
        title=title,
        dragmode="pan",
        xaxis_title="Number of AMR publications",
        yaxis=dict(
            title="Scientific interest (signed log10 scale)",
            tickmode="array",
            tickvals=si_tick_positions.tolist(),
            ticktext=si_tick_text,
        ),
        template="plotly_white",
        width=int(width),
        height=int(height),
    )
    fig.update_xaxes(type="log")
    if show:
        _configure_plotly_inline_renderer(pio)
        fig.show()
    return fig


def plot_interactive_latest_observed_count_vs_future_si_shift(
    rank_df: pd.DataFrame,
    *,
    GLASS_pairs_df: pd.DataFrame | None = None,
    min_last_available_count: float = 10.0,
    min_counts_background: float = 100.0,
    min_count_abs_pct_error_mean: float = 0.0,
    max_count_abs_pct_error_mean: float = 10.0,
    min_abs_delta_si: float = 0.05,
    min_si_delta: float | None = None,
    sample_n: int | None = None,
    random_state: int | None = None,
    show: bool = True,
    display_name: bool = False,
    title: str | None = None,
    width: int = 1400,
    height: int = 900,
    marker_color: str = "#1f77b4",
    pdf_path: str | None = None,
) -> dict[str, Any]:
    """Plot latest available AMR count against future-minus-latest-smoothed SI shift using precomputed rank rows."""
    def _signed_log10(values: pd.Series | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.sign(arr) * np.log10(1.0 + np.abs(arr))

    def _normalize_text(values: pd.Series) -> pd.Series:
        return values.astype(str).str.strip()

    required_cols = {
        "pathogen",
        "antibiotic",
        "latest_available_year",
        "future_year",
        "latest_available_count",
        "latest_available_predicted_SI",
        "future_predicted_SI",
        "post_fit_count_abs_pct_error_mean",
        "n_count_error_years",
        "status",
    }
    missing = required_cols - set(rank_df.columns)
    if missing:
        raise KeyError(f"rank_df is missing required columns: {sorted(missing)}")

    try:
        import plotly.io as pio
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional notebook dependency
        raise ImportError("plotly is required for interactive SI plots.") from exc

    plot_df = rank_df.copy()
    plot_df = plot_df.loc[plot_df["status"].eq("ok")].copy()
    if plot_df.empty:
        raise ValueError("No successful ranking rows are available for plotting.")

    plot_df = plot_df.dropna(
        subset=[
            "latest_available_count",
            "latest_available_predicted_SI",
            "future_predicted_SI",
            "post_fit_count_abs_pct_error_mean",
        ]
    )
    plot_df = plot_df.loc[
        (pd.to_numeric(plot_df["latest_available_count"], errors="coerce") >= float(min_last_available_count))
        & (pd.to_numeric(plot_df["post_fit_count_abs_pct_error_mean"], errors="coerce") >= float(min_count_abs_pct_error_mean))
        & (pd.to_numeric(plot_df["post_fit_count_abs_pct_error_mean"], errors="coerce") <= float(max_count_abs_pct_error_mean))
    ].reset_index(drop=True)
    if plot_df.empty:
        raise ValueError(
            "No rows remain after filtering by minimum last-available count and mean count percentage error bounds."
        )

    if sample_n is not None and int(sample_n) > 0 and len(plot_df) > int(sample_n):
        plot_df = plot_df.sample(int(sample_n), random_state=random_state).reset_index(drop=True)

    plot_df["latest_smoothed_SI"] = pd.to_numeric(plot_df["latest_available_predicted_SI"], errors="coerce")
    plot_df["future_minus_latest_smoothed_SI"] = (
        pd.to_numeric(plot_df["future_predicted_SI"], errors="coerce")
        - plot_df["latest_smoothed_SI"]
    )
    if min_si_delta is not None:
        plot_df = plot_df.loc[
            pd.to_numeric(plot_df["future_minus_latest_smoothed_SI"], errors="coerce") >= float(min_si_delta)
        ].reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("No rows remain after applying the displayed SI-delta filter.")

    plot_df["is_glass_pair"] = False
    plot_df["glass_group_column"] = None
    if GLASS_pairs_df is not None and not GLASS_pairs_df.empty:
        glass_required_cols = {"pathogen"}
        missing_glass = glass_required_cols - set(GLASS_pairs_df.columns)
        if missing_glass:
            raise KeyError(f"GLASS_pairs_df is missing required columns: {sorted(missing_glass)}")

        glass_work = GLASS_pairs_df.copy()
        glass_work["pathogen"] = _normalize_text(glass_work["pathogen"])

        rank_antibiotics = set(_normalize_text(plot_df["antibiotic"]).dropna().unique())
        glass_candidate_cols = [
            col
            for col in ["antibiotic", "Class", "Simplified categories", "High level categories"]
            if col in glass_work.columns
        ]
        selected_glass_col = None
        best_overlap = -1
        for candidate_col in glass_candidate_cols:
            candidate_values = set(_normalize_text(glass_work[candidate_col].dropna()).unique())
            overlap = len(rank_antibiotics & candidate_values)
            if overlap > best_overlap:
                best_overlap = overlap
                selected_glass_col = candidate_col

        if selected_glass_col is not None and best_overlap > 0:
            glass_pairs = glass_work[["pathogen", selected_glass_col]].copy()
            glass_pairs = glass_pairs.dropna(subset=[selected_glass_col]).drop_duplicates()
            glass_pairs = glass_pairs.rename(columns={selected_glass_col: "antibiotic"})
            glass_pairs["pathogen"] = _normalize_text(glass_pairs["pathogen"])
            glass_pairs["antibiotic"] = _normalize_text(glass_pairs["antibiotic"])

            plot_pairs = pd.DataFrame(
                {
                    "pathogen": _normalize_text(plot_df["pathogen"]),
                    "antibiotic": _normalize_text(plot_df["antibiotic"]),
                }
            )
            plot_df["is_glass_pair"] = plot_pairs.merge(
                glass_pairs.assign(is_glass_pair=True),
                on=["pathogen", "antibiotic"],
                how="left",
            )["is_glass_pair"].fillna(False).astype(bool)
            plot_df["glass_group_column"] = selected_glass_col
    plot_df["appears_in_GLASS"] = np.where(plot_df["is_glass_pair"], "yes", "no")

    plot_df["label"] = plot_df.apply(
        lambda row: f"{str(row['pathogen']).strip()[:3]}<br>{str(row['antibiotic']).strip()[:3]}",
        axis=1,
    )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=plot_df["latest_available_count"],
                y=plot_df["future_minus_latest_smoothed_SI"],
                mode="markers+text" if display_name else "markers",
                text=plot_df["label"] if display_name else None,
                textposition="top center" if display_name else None,
                marker=dict(
                    color=pd.to_numeric(plot_df["post_fit_count_abs_pct_error_mean"], errors="coerce"),
                    colorscale=[[0.0, "#b30000"], [1.0, "#ffffff"]],
                    cmin=0.0,
                    cmax=float(max_count_abs_pct_error_mean),
                    size=13,
                    opacity=0.8,
                    line=dict(
                        color=["#000000" if is_glass else "#000000" for is_glass in plot_df["is_glass_pair"]],
                        width=[3.8 if is_glass else 1.2 for is_glass in plot_df["is_glass_pair"]],
                    ),
                    colorbar=dict(title="Mean abs count % error"),
                ),
                customdata=plot_df[
                    [
                        "pathogen",
                        "antibiotic",
                        "latest_available_year",
                        "future_year",
                        "latest_available_count",
                        "latest_smoothed_SI",
                        "future_predicted_SI",
                        "latest_available_count",
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
                    "Counts: %{customdata[4]:.0f} -> %{customdata[8]:.0f}<br>"
                    "Smoothed SI: %{customdata[5]:.3f} -> %{customdata[6]:.3f}<br>"
                    "Delta SI: %{customdata[9]:.3f}<br>"
                    "Error (pc): %{customdata[10]:.2f}<br>"
                    "GLASS: %{customdata[11]}<extra></extra>"
                ),
                name="Pairs",
            )
        ]
    )

    if title is None:
        title = (
            f"Latest available AMR count vs predicted SI shift from latest smoothed SI "
            f"(mean abs count error {min_count_abs_pct_error_mean:g}% to {max_count_abs_pct_error_mean:g}%)"
        )

    x_min = float(pd.to_numeric(plot_df["latest_available_count"], errors="coerce").min())
    x_max = float(pd.to_numeric(plot_df["latest_available_count"], errors="coerce").max())
    background_threshold = float(min_counts_background)
    x_axis_min = max(min(x_min, background_threshold) / 1.15, 1e-3)
    x_axis_max = max(x_max * 1.08, x_axis_min * 1.01)

    y_min = float(pd.to_numeric(plot_df["future_minus_latest_smoothed_SI"], errors="coerce").min())
    y_max = float(pd.to_numeric(plot_df["future_minus_latest_smoothed_SI"], errors="coerce").max())
    y_span = max(y_max - y_min, 1e-6)
    y_axis_min = y_min - 0.08 * y_span
    y_axis_max = y_max + 0.08 * y_span

    fig.update_layout(
        title=title,
        dragmode="pan",
        xaxis_title="AMR publications in latest available year",
        yaxis_title="Predicted future SI - smoothed SI at latest available year",
        template="plotly_white",
        width=int(width),
        height=int(height),
        margin=dict(l=150, r=180, t=110, b=95),
    )
    fig.update_xaxes(type="log", range=[np.log10(x_axis_min), np.log10(x_axis_max)])
    fig.update_yaxes(range=[y_axis_min, y_axis_max])

    if x_axis_min < background_threshold:
        fig.add_shape(
            type="rect",
            x0=x_axis_min,
            x1=min(background_threshold, x_axis_max),
            xref="x",
            fillcolor="#f3f3f3",
            opacity=1.0,
            line_width=0,
            layer="below",
            yref="paper",
            y0=0.0,
            y1=1.0,
        )
    if background_threshold < x_axis_max:
        fig.add_shape(
            type="rect",
            x0=max(background_threshold, x_axis_min),
            x1=x_axis_max,
            xref="x",
            fillcolor="#ffffff",
            opacity=1.0,
            line_width=0,
            layer="below",
            yref="paper",
            y0=0.0,
            y1=1.0,
        )
    delta_si = abs(float(min_abs_delta_si))
    if delta_si > 0.0:
        fig.add_shape(
            type="rect",
            y0=max(-delta_si, y_axis_min),
            y1=min(delta_si, y_axis_max),
            yref="y",
            fillcolor="rgba(160, 160, 160, 0.22)",
            opacity=1.0,
            line_width=0,
            layer="below",
            xref="paper",
            x0=0.0,
            x1=1.0,
        )
        fig.add_hline(y=-delta_si, line_dash="dot", line_color="gray", opacity=0.7)
        fig.add_hline(y=delta_si, line_dash="dot", line_color="gray", opacity=0.7)
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", opacity=0.7)

    resolved_pdf_path = os.path.abspath(pdf_path or "latest_observed_count_vs_future_si_shift.pdf")
    try:
        fig.write_image(resolved_pdf_path)
    except Exception as exc:  # pragma: no cover - optional notebook dependency
        raise RuntimeError(
            "Failed to save Plotly figure as PDF. Ensure the Plotly static image backend is installed."
        ) from exc
    print(resolved_pdf_path)

    if show:
        _configure_plotly_inline_renderer(pio)
        fig.show()

    return {"figure": fig, "plot_df": plot_df, "pdf_path": resolved_pdf_path}


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
    "rank_yearly_pairs_by_future_si_shift",
    "select_interesting_yearly_rank_cases",
    "plot_si_prediction",
    "plot_yearly_si_pair",
    "display_si_amr_pareto_front",
    "plot_si_prediction_for_pareto_pairs",
    "plot_yearly_si_prediction_for_pareto_pairs",
    "plot_interactive_yearly_rank_shift",
    "plot_interactive_latest_observed_count_vs_future_si_shift",
    "get_pair_si_series",
    "fit_predict_si_gp",
    "forecast_si_for_all_pairs",
]
