"""Assumption-explicit panel estimates for frozen SI concept catalogs.

These estimates are causal only under the documented conditional-exchangeability,
consistency, and positivity assumptions. The default available covariates do not
include external AMR burden or policy measurements, so reports must label them
as exploratory rather than identified effects.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


_CAUSAL_INTERPRETATION = (
    "Exploratory counterfactual attribution; causal interpretation requires no unmeasured "
    "time-varying confounding, adequate positivity, and a well-defined intervention on concept prevalence."
)


def estimate_pair_concept_causal_rankings(
    si_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    catalog: Sequence[Mapping[str, Any]],
    pair_split: pd.DataFrame,
    *,
    fit_start_year: int = 2000,
    fit_end_year: int = 2023,
    horizons: Sequence[int] = (1, 2, 3),
    lag_length: int = 3,
    min_prior_observations: int = 10,
    min_pair_origins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate development-fitted, pair-specific counterfactual concept contributions.

    For each direct horizon, a pooled linear panel model includes pair and origin-year
    fixed effects, lagged SI, lagged abstract volume, and lagged signals for every
    frozen concept. It is fit on development pairs only. Pair-specific contributions
    contrast the pair's observed lagged concept history with zero concept signal.
    This is an assumption-dependent g-computation summary, not proof of causation.

    Returns a long horizon table and an equal-weight aggregate ranking (horizon 0).
    """
    required_si = {"pathogen", "antibiotic", "year", "SI"}
    required_signals = {"pathogen", "antibiotic", "year", "concept_id", "concept_prevalence"}
    required_corpus = {"pathogen", "antibiotic", "year", "pmid"}
    required_split = {"pathogen", "antibiotic", "split"}
    for frame, required, name in (
        (si_df, required_si, "si_df"), (signals_df, required_signals, "signals_df"),
        (corpus_df, required_corpus, "corpus_df"), (pair_split, required_split, "pair_split"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} is missing columns: {sorted(missing)}")
    if lag_length < 1 or not horizons or any(int(horizon) < 1 for horizon in horizons):
        raise ValueError("lag_length and forecast horizons must be positive.")

    concept_ids = [str(concept["concept_id"]) for concept in catalog]
    concept_names = {str(c["concept_id"]): str(c.get("name", c["concept_id"])) for c in catalog}
    si = si_df[["pathogen", "antibiotic", "year", "SI"]].copy()
    si["pathogen"] = si["pathogen"].astype(str)
    si["antibiotic"] = si["antibiotic"].astype(str)
    si["year"] = pd.to_numeric(si["year"], errors="coerce")
    si["SI"] = pd.to_numeric(si["SI"], errors="coerce")
    si = si.dropna(subset=["year", "SI"])
    si["year"] = si["year"].astype(int)
    si = si.drop_duplicates(["pathogen", "antibiotic", "year"], keep="last")
    y_lookup = {
        (str(r.pathogen), str(r.antibiotic), int(r.year)): float(r.SI)
        for r in si.itertuples(index=False)
    }
    pair_years = {
        (str(pair[0]), str(pair[1])): np.sort(rows["year"].to_numpy(dtype=int))
        for pair, rows in si.groupby(["pathogen", "antibiotic"], sort=False)
    }
    observed_years = {
        pair: np.sort(rows["year"].to_numpy(dtype=int))
        for pair, rows in si.groupby(["pathogen", "antibiotic"], sort=False)
    }

    signals = signals_df.copy()
    signals["pathogen"] = signals["pathogen"].astype(str)
    signals["antibiotic"] = signals["antibiotic"].astype(str)
    signals["year"] = pd.to_numeric(signals["year"], errors="coerce")
    signals["concept_id"] = signals["concept_id"].astype(str)
    x_lookup = {
        (str(r.pathogen), str(r.antibiotic), int(r.year), str(r.concept_id)):
        float(r.concept_prevalence) if pd.notna(r.concept_prevalence) else 0.0
        for r in signals.dropna(subset=["year"]).itertuples(index=False)
    }
    pub = corpus_df[["pathogen", "antibiotic", "year", "pmid"]].copy()
    pub["pathogen"] = pub["pathogen"].astype(str)
    pub["antibiotic"] = pub["antibiotic"].astype(str)
    pub["year"] = pd.to_numeric(pub["year"], errors="coerce")
    pub = pub.dropna(subset=["year"])
    pub["year"] = pub["year"].astype(int)
    pub_counts = pub.groupby(["pathogen", "antibiotic", "year"])["pmid"].nunique().to_dict()

    feature_cols = (
        [f"si_lag_{lag}" for lag in range(1, lag_length + 1)]
        + [f"log_abstracts_lag_{lag}" for lag in range(1, lag_length + 1)]
        + [f"{concept_id}_lag_{lag}" for concept_id in concept_ids for lag in range(1, lag_length + 1)]
    )
    rows_by_horizon: dict[int, list[dict[str, Any]]] = {int(h): [] for h in horizons}
    for (pathogen, antibiotic), years in pair_years.items():
        for origin in years:
            if origin < fit_start_year or origin > fit_end_year:
                continue
            prior_count = int(np.searchsorted(observed_years[(pathogen, antibiotic)], origin, side="left"))
            if prior_count < min_prior_observations:
                continue
            for horizon in map(int, horizons):
                target_year = int(origin) + horizon
                if (pathogen, antibiotic, target_year) not in y_lookup:
                    continue
                values: dict[str, float] = {}
                complete_history = True
                for lag in range(1, lag_length + 1):
                    history_year = int(origin) - lag
                    si_value = y_lookup.get((pathogen, antibiotic, history_year))
                    if si_value is None:
                        complete_history = False
                        break
                    values[f"si_lag_{lag}"] = si_value
                    values[f"log_abstracts_lag_{lag}"] = float(np.log1p(
                        pub_counts.get((pathogen, antibiotic, history_year), 0)
                    ))
                    for concept_id in concept_ids:
                        values[f"{concept_id}_lag_{lag}"] = x_lookup.get(
                            (pathogen, antibiotic, history_year, concept_id), 0.0
                        )
                if complete_history:
                    rows_by_horizon[horizon].append({
                        "pathogen": pathogen, "antibiotic": antibiotic,
                        "pair_key": f"{pathogen}\x1f{antibiotic}",
                        "origin_year": int(origin), "target_year": target_year,
                        "target": y_lookup[(pathogen, antibiotic, target_year)], **values,
                    })

    split = pair_split.copy()
    split[["pathogen", "antibiotic"]] = split[["pathogen", "antibiotic"]].astype(str)
    split_lookup = {
        (str(r.pathogen), str(r.antibiotic)): str(r.split)
        for r in split.itertuples(index=False)
    }
    development_pairs = {pair for pair, split_name in split_lookup.items() if split_name == "development"}
    estimate_rows: list[dict[str, Any]] = []
    for horizon, rows in rows_by_horizon.items():
        panel = pd.DataFrame(rows)
        if panel.empty:
            continue
        train = panel.loc[
            [(row.pathogen, row.antibiotic) in development_pairs for row in panel.itertuples(index=False)]
        ].copy()
        if train.empty:
            continue
        pair_fe = pd.get_dummies(train["pair_key"], prefix="pair", drop_first=True, dtype=float)
        year_fe = pd.get_dummies(train["origin_year"].astype(str), prefix="year", drop_first=True, dtype=float)
        numerical_all = train[feature_cols].astype(float)
        # Concepts absent from development pairs create all-zero columns and should
        # not make an otherwise usable shared model appear rank deficient.
        active_feature_cols = [
            column for column in feature_cols
            if float(numerical_all[column].std(ddof=0)) > 1e-10
        ]
        numerical = numerical_all[active_feature_cols]
        design_frame = pd.concat(
            [pd.Series(1.0, index=train.index, name="intercept"), pair_fe, year_fe, numerical], axis=1
        )
        design = design_frame.to_numpy(dtype=float)
        target = train["target"].to_numpy(dtype=float)
        if (len(train) <= design.shape[1] or train["pair_key"].nunique() < 2
                or np.linalg.matrix_rank(design) < design.shape[1]):
            for (pathogen, antibiotic), pair_rows in panel.groupby(["pathogen", "antibiotic"], sort=False):
                for concept_id in concept_ids:
                    estimate_rows.append({
                        "pathogen": str(pathogen), "antibiotic": str(antibiotic), "concept_id": concept_id,
                        "split": split_lookup.get((str(pathogen), str(antibiotic)), "unassigned"),
                        "concept_name": concept_names[concept_id], "horizon": horizon,
                        "status": "development_model_not_identified", "n_pair_origins": len(pair_rows),
                        "causal_interpretation": _CAUSAL_INTERPRETATION,
                    })
            continue
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ coefficients
        bread = np.linalg.pinv(design.T @ design, hermitian=True)
        meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
        train_groups = train["pair_key"].to_numpy()
        unique_groups = np.unique(train_groups)
        for group in unique_groups:
            indexes = np.flatnonzero(train_groups == group)
            score = design[indexes].T @ residual[indexes]
            meat += np.outer(score, score)
        correction = 1.0
        if len(unique_groups) > 1 and len(train) > design.shape[1]:
            correction = (len(unique_groups) / (len(unique_groups) - 1)) * (
                (len(train) - 1) / (len(train) - design.shape[1])
            )
        covariance = correction * bread @ meat @ bread
        feature_index = {name: design_frame.columns.get_loc(name) for name in active_feature_cols}

        for (pathogen, antibiotic), pair_rows in panel.groupby(["pathogen", "antibiotic"], sort=False):
            pair_rows = pair_rows.sort_values("origin_year")
            pair_key = (str(pathogen), str(antibiotic))
            n_origins = int(pair_rows["origin_year"].nunique())
            for concept_id in concept_ids:
                names = [f"{concept_id}_lag_{lag}" for lag in range(1, lag_length + 1)]
                history = pair_rows[names].to_numpy(dtype=float)
                exposure_support = bool(history.size and np.nanmax(history) > 0 and np.nanstd(history) > 1e-8)
                weights = np.zeros(design.shape[1], dtype=float)
                for feature in names:
                    if feature in feature_index:
                        weights[feature_index[feature]] = float(pair_rows[feature].mean())
                contribution = float(weights @ coefficients)
                contribution_se = float(np.sqrt(max(0.0, weights @ covariance @ weights)))
                intervention_weights = np.zeros(design.shape[1], dtype=float)
                for feature in names:
                    if feature in feature_index:
                        intervention_weights[feature_index[feature]] = 0.1
                effect_per_10pp = float(intervention_weights @ coefficients)
                effect_se = float(np.sqrt(max(0.0, intervention_weights @ covariance @ intervention_weights)))
                if n_origins < min_pair_origins:
                    status = "insufficient_pair_origins"
                elif not exposure_support:
                    status = "no_within_pair_exposure_variation"
                elif any(feature not in feature_index for feature in names):
                    status = "limited_development_lag_support"
                else:
                    status = "assumption_dependent"
                estimate_rows.append({
                    "pathogen": pair_key[0], "antibiotic": pair_key[1],
                    "split": split_lookup.get(pair_key, "unassigned"),
                    "concept_id": concept_id, "concept_name": concept_names[concept_id],
                    "horizon": horizon,
                    "historical_contribution": contribution,
                    "historical_contribution_ci_lower": contribution - 1.96 * contribution_se,
                    "historical_contribution_ci_upper": contribution + 1.96 * contribution_se,
                    "effect_per_10_percentage_points": effect_per_10pp,
                    "effect_per_10pp_ci_lower": effect_per_10pp - 1.96 * effect_se,
                    "effect_per_10pp_ci_upper": effect_per_10pp + 1.96 * effect_se,
                    "n_pair_origins": n_origins,
                    "n_positive_exposure_origins": int((history.sum(axis=1) > 0).sum()),
                    "status": status,
                    "causal_interpretation": _CAUSAL_INTERPRETATION,
                })

    horizons_df = pd.DataFrame(estimate_rows)
    # Keep a row for every split pair and catalog concept, including pairs that
    # never reach the minimum-history threshold or have no usable target origins.
    existing = set()
    if not horizons_df.empty:
        existing = set(zip(
            horizons_df["pathogen"].astype(str), horizons_df["antibiotic"].astype(str),
            horizons_df["concept_id"].astype(str), horizons_df["horizon"].astype(int),
        ))
    missing_rows = []
    for (pathogen, antibiotic), split_name in split_lookup.items():
        years = pair_years.get((pathogen, antibiotic), np.asarray([], dtype=int))
        for concept_id in concept_ids:
            for horizon in map(int, horizons):
                key = (pathogen, antibiotic, concept_id, horizon)
                if key in existing:
                    continue
                missing_rows.append({
                    "pathogen": pathogen, "antibiotic": antibiotic, "split": split_name,
                    "concept_id": concept_id, "concept_name": concept_names[concept_id],
                    "horizon": horizon, "n_pair_origins": 0,
                    "status": "insufficient_pair_history" if len(years) < min_prior_observations else "no_usable_forecast_origins",
                    "causal_interpretation": _CAUSAL_INTERPRETATION,
                })
    if missing_rows:
        horizons_df = pd.concat([horizons_df, pd.DataFrame(missing_rows)], ignore_index=True)
    if horizons_df.empty:
        return horizons_df, horizons_df.copy()
    valid_horizons = list(map(int, horizons))
    aggregate_rows = []
    for (pathogen, antibiotic, concept_id), group in horizons_df.groupby(
        ["pathogen", "antibiotic", "concept_id"], sort=False
    ):
        valid = group.loc[group["status"].eq("assumption_dependent")]
        complete = set(valid["horizon"].astype(int)) == set(valid_horizons)
        if complete:
            contribution = float(valid["historical_contribution"].mean())
            effect_per_10pp = float(valid["effect_per_10_percentage_points"].mean())
            status = "assumption_dependent"
            n_origins = int(valid["n_pair_origins"].min())
            positive_origins = int(valid["n_positive_exposure_origins"].max())
        else:
            contribution = np.nan
            effect_per_10pp = np.nan
            status = "incomplete_horizons"
            n_origins = int(group["n_pair_origins"].min()) if "n_pair_origins" in group else 0
            positive_origins = int(group["n_positive_exposure_origins"].max()) if "n_positive_exposure_origins" in group else 0
        first = group.iloc[0]
        aggregate_rows.append({
            "pathogen": pathogen, "antibiotic": antibiotic,
            "split": first.get("split", split_lookup.get((str(pathogen), str(antibiotic)), "unassigned")),
            "concept_id": concept_id, "concept_name": first["concept_name"], "horizon": 0,
            "historical_contribution": contribution,
            "historical_contribution_ci_lower": np.nan,
            "historical_contribution_ci_upper": np.nan,
            "effect_per_10_percentage_points": effect_per_10pp,
            "effect_per_10pp_ci_lower": np.nan, "effect_per_10pp_ci_upper": np.nan,
            "n_pair_origins": n_origins,
            "n_positive_exposure_origins": positive_origins,
            "status": status,
            "causal_interpretation": _CAUSAL_INTERPRETATION,
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["causal_rank"] = pd.Series(pd.NA, index=aggregate.index, dtype="Int64")
    for _, indexes in aggregate.groupby(["pathogen", "antibiotic"], sort=False).groups.items():
        idx = list(indexes)
        supported = aggregate.loc[idx].loc[aggregate.loc[idx, "status"].eq("assumption_dependent")]
        order = supported["historical_contribution"].abs().sort_values(ascending=False).index
        aggregate.loc[order, "causal_rank"] = range(1, len(order) + 1)
    aggregate["effect_direction"] = np.select(
        [aggregate["historical_contribution"].gt(0), aggregate["historical_contribution"].lt(0)],
        ["higher SI", "lower SI"], default="not estimated",
    )
    return horizons_df, aggregate
