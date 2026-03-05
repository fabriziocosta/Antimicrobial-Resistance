"""Predictive modeling helpers for AMR notebooks."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


def _should_apply_log1p(value_col: str, apply_log1p: bool | None) -> bool:
    """Use log1p for count-like signals by default; keep score-like signals raw."""
    if apply_log1p is None:
        return value_col == "count"
    return bool(apply_log1p)


def _prepare_amr_df(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Validate and normalize AMR dataframe fields used by predictive helpers."""
    out = df.copy()
    if value_col not in out.columns:
        available = ", ".join(sorted(out.columns))
        raise ValueError(f"Column '{value_col}' not found. Available columns: {available}")
    out["start_date"] = pd.to_datetime(out["start_date"])
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[value_col])
    return out


def _normalize_level_name(level: str) -> str:
    return str(level).strip().lower().replace(" ", "_")


def _resolve_antibiotic_level_column(level: str) -> str:
    """Map user-facing level names to metadata column names."""
    level_map = {
        "antibiotic": "Antibiotic",
        "class": "Class",
        "simplified_category": "Simplified categories",
        "simplified_categories": "Simplified categories",
        "high_level_category": "High level categories",
        "high_level_categories": "High level categories",
        "origin": "Origin",
    }
    key = _normalize_level_name(level)
    return level_map.get(key, level)


def _apply_antibiotic_partition(
    df: pd.DataFrame,
    antibiotic_level: str = "antibiotic",
    antibiotic_meta_df: pd.DataFrame | None = None,
    error_on_unmapped: bool = True,
) -> tuple[pd.DataFrame, str]:
    """
    Add an antibiotic partition column used for model aggregation.
    Returns dataframe and partition column name.
    """
    out = df.copy()
    partition_col = "_antibiotic_partition"
    level_key = _normalize_level_name(antibiotic_level)
    if level_key == "antibiotic":
        out[partition_col] = out["antibiotic"].astype(str)
        return out, partition_col

    if antibiotic_meta_df is None:
        raise ValueError(
            f"antibiotic_level='{antibiotic_level}' requires antibiotic_meta_df mapping."
        )

    meta = antibiotic_meta_df.copy()
    if "Antibiotic" not in meta.columns:
        available = ", ".join(sorted(meta.columns))
        raise ValueError(
            "antibiotic_meta_df must include an 'Antibiotic' column. "
            f"Available columns: {available}"
        )

    level_col = _resolve_antibiotic_level_column(antibiotic_level)
    if level_col not in meta.columns:
        available = ", ".join(sorted(meta.columns))
        raise ValueError(
            f"Requested antibiotic_level '{antibiotic_level}' not found in antibiotic_meta_df. "
            f"Expected metadata column '{level_col}'. Available columns: {available}"
        )

    meta = meta[["Antibiotic", level_col]].copy()
    meta["Antibiotic"] = meta["Antibiotic"].astype(str)
    meta[level_col] = meta[level_col].astype(str)
    meta = meta.drop_duplicates(subset=["Antibiotic"])
    mapper = dict(zip(meta["Antibiotic"], meta[level_col]))

    out["antibiotic"] = out["antibiotic"].astype(str)
    mapped = out["antibiotic"].map(mapper)
    if error_on_unmapped:
        missing = sorted(out.loc[mapped.isna(), "antibiotic"].unique().tolist())
        if missing:
            preview = ", ".join(missing[:20])
            extra = "" if len(missing) <= 20 else f" ... (+{len(missing)-20} more)"
            raise ValueError(
                f"{len(missing)} antibiotics are missing from metadata for level '{antibiotic_level}': "
                f"{preview}{extra}"
            )
    else:
        if mapped.isna().any():
            n_missing = int(mapped.isna().sum())
            warnings.warn(
                f"{n_missing} antibiotic rows are unmapped for level '{antibiotic_level}' and will keep original label.",
                RuntimeWarning,
            )
    out[partition_col] = mapped.fillna(out["antibiotic"]).astype(str)
    return out, partition_col


def _split_feature_target_by_bin(
    df: pd.DataFrame,
    value_col: str,
    use_log1p: bool,
    target_bin,
    partition_col: str = "antibiotic",
):
    """Build feature/target matrices for a specific target period."""
    all_bins = sorted(df["start_date"].unique())
    if len(all_bins) < 2:
        raise ValueError("At least two time bins are required to build features and target.")
    if target_bin not in all_bins:
        raise ValueError(f"target_bin {target_bin} is not present in data.")

    feature_bins = [b for b in all_bins if b < target_bin]
    if not feature_bins:
        raise ValueError("No feature bins exist before target_bin.")

    feature_df = df[df["start_date"].isin(feature_bins)].pivot_table(
        index="pathogen",
        columns=[partition_col, "start_date"],
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    )
    if use_log1p:
        feature_df = np.log1p(feature_df)
    feature_df.columns = [f"{antibiotic}_{start.date()}" for antibiotic, start in feature_df.columns]

    target_df = df[df["start_date"] == target_bin].pivot_table(
        index="pathogen",
        columns=partition_col,
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    )
    if use_log1p:
        target_df = np.log1p(target_df)
    target_df.columns = [f"{antibiotic}_target" for antibiotic in target_df.columns]

    # Keep only pathogens present in both feature and target matrices.
    common_pathogens = feature_df.index.intersection(target_df.index)
    feature_df = feature_df.loc[common_pathogens]
    target_df = target_df.loc[common_pathogens]
    if feature_df.empty or target_df.empty:
        raise ValueError("No shared pathogens between feature bins and target bin.")

    return feature_df, target_df


def _antibiotics_from_target_columns(target_cols) -> list[str]:
    return [str(col).replace("_target", "") for col in target_cols]


def evaluate_predictions(Y_true: np.ndarray, Y_pred: np.ndarray, antibiotics: list[str] | None = None):
    """Return aggregate and per-antibiotic metrics for multi-output regression."""
    overall = {
        "R2": float(skmetrics.r2_score(Y_true, Y_pred, multioutput="uniform_average")),
        "RMSE": float(np.sqrt(skmetrics.mean_squared_error(Y_true, Y_pred))),
        "MAE": float(skmetrics.mean_absolute_error(Y_true, Y_pred)),
    }

    n_targets = Y_true.shape[1]
    if antibiotics is None:
        antibiotics = [f"target_{i}" for i in range(n_targets)]
    antibiotics = [str(a) for a in antibiotics]

    rows = []
    for j, ab in enumerate(antibiotics):
        yt = Y_true[:, j]
        yp = Y_pred[:, j]
        rmse_j = float(np.sqrt(skmetrics.mean_squared_error(yt, yp)))
        mae_j = float(skmetrics.mean_absolute_error(yt, yp))
        if np.nanstd(yt) == 0:
            r2_j = np.nan
        else:
            r2_j = float(skmetrics.r2_score(yt, yp))
        rows.append({"antibiotic": ab, "R2": r2_j, "RMSE": rmse_j, "MAE": mae_j})
    per_antibiotic = pd.DataFrame(rows)
    return overall, per_antibiotic


def build_taxonomy_onehot(df: pd.DataFrame, taxonomy_cols=None):
    """Build one-hot embedding vectors for taxonomic information."""
    if taxonomy_cols is None:
        taxonomy_cols = ["domain", "kingdom", "phylum", "class", "order", "family"]

    df_onehot = pd.get_dummies(df[taxonomy_cols], prefix=taxonomy_cols)
    embeddings = df_onehot.values.astype(int).tolist()

    df = df.copy()
    df["taxonomy_embeddings"] = embeddings
    return df, df_onehot


def build_data_and_target(
    df: pd.DataFrame,
    value_col: str = "count",
    apply_log1p: bool | None = None,
    target_bin=None,
    antibiotic_level: str = "antibiotic",
    antibiotic_meta_df: pd.DataFrame | None = None,
    error_on_unmapped: bool = True,
):
    """Build feature (X) and target (Y) matrices for pathogens."""
    df = _prepare_amr_df(df, value_col=value_col)
    df, partition_col = _apply_antibiotic_partition(
        df,
        antibiotic_level=antibiotic_level,
        antibiotic_meta_df=antibiotic_meta_df,
        error_on_unmapped=error_on_unmapped,
    )
    use_log1p = _should_apply_log1p(value_col, apply_log1p)
    all_bins = sorted(df["start_date"].unique())
    if len(all_bins) < 2:
        raise ValueError("At least two time bins are required to build features and target.")
    target_bin = all_bins[-1] if target_bin is None else pd.to_datetime(target_bin)
    feature_df, target_df = _split_feature_target_by_bin(
        df=df,
        value_col=value_col,
        use_log1p=use_log1p,
        target_bin=target_bin,
        partition_col=partition_col,
    )

    X = feature_df.to_numpy()
    Y = target_df.to_numpy()

    return X, Y, feature_df, target_df


def build_amr_with_embeddings(
    amr_df: pd.DataFrame,
    data_df: pd.DataFrame,
    embedding_cols,
    value_col: str = "count",
    apply_log1p: bool | None = None,
    target_bin=None,
    antibiotic_level: str = "antibiotic",
    antibiotic_meta_df: pd.DataFrame | None = None,
    error_on_unmapped: bool = True,
):
    """Build AMR features/targets and concatenate pathogen embeddings."""
    X_amr, Y, feature_df, target_df = build_data_and_target(
        amr_df,
        value_col=value_col,
        apply_log1p=apply_log1p,
        target_bin=target_bin,
        antibiotic_level=antibiotic_level,
        antibiotic_meta_df=antibiotic_meta_df,
        error_on_unmapped=error_on_unmapped,
    )

    pathogens = feature_df.index
    data_sub = data_df.set_index("CanonicalSpecies").loc[pathogens]

    embeddings_all = []
    for col in embedding_cols:
        emb = np.vstack(data_sub[col].apply(np.array).values)
        embeddings_all.append(emb)

    embeddings_aligned = np.hstack(embeddings_all)
    X = np.hstack([X_amr, embeddings_aligned])
    return X, Y, feature_df, target_df, embeddings_aligned


def _compute_sample_weights(
    amr_df: pd.DataFrame,
    pathogens,
    value_col: str,
    sample_weight_mode: str = "count_if_available",
) -> np.ndarray | None:
    mode = str(sample_weight_mode).strip().lower()
    if mode in {"none", "off", "no"}:
        return None

    work = amr_df.copy()
    if mode in {"count_if_available", "count"} and "count" in work.columns:
        totals = pd.to_numeric(work["count"], errors="coerce")
        work = work.assign(_w=totals).dropna(subset=["_w"])
        weights = work.groupby("pathogen")["_w"].sum()
    elif mode in {"value_abs", "abs_value", "value"}:
        vals = pd.to_numeric(work[value_col], errors="coerce")
        work = work.assign(_w=vals.abs()).dropna(subset=["_w"])
        weights = work.groupby("pathogen")["_w"].sum()
    elif mode in {"count_if_available", "count"}:
        vals = pd.to_numeric(work[value_col], errors="coerce")
        work = work.assign(_w=vals.abs()).dropna(subset=["_w"])
        weights = work.groupby("pathogen")["_w"].sum()
    else:
        raise ValueError(
            f"Unsupported sample_weight_mode '{sample_weight_mode}'. "
            "Use one of: count_if_available, count, value_abs, none."
        )
    return np.log1p(weights.reindex(pathogens).fillna(0).values)


def random_forest_experiment_with_embeddings(
    amr_df: pd.DataFrame,
    data_df: pd.DataFrame,
    embedding_cols=None,
    value_col: str = "count",
    apply_log1p: bool | None = None,
    n_repeats: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth=None,
    target_bin=None,
    antibiotic_level: str = "antibiotic",
    antibiotic_meta_df: pd.DataFrame | None = None,
    error_on_unmapped: bool = True,
    sample_weight_mode: str = "count_if_available",
):
    """Run repeated train/test experiments with RandomForest on pathogen AMR features."""
    use_log1p = _should_apply_log1p(value_col, apply_log1p)
    if embedding_cols:
        X, Y, feature_df, target_df, embeddings = build_amr_with_embeddings(
            amr_df,
            data_df,
            embedding_cols,
            value_col=value_col,
            apply_log1p=use_log1p,
            target_bin=target_bin,
            antibiotic_level=antibiotic_level,
            antibiotic_meta_df=antibiotic_meta_df,
            error_on_unmapped=error_on_unmapped,
        )
    else:
        X, Y, feature_df, target_df = build_data_and_target(
            amr_df,
            value_col=value_col,
            apply_log1p=use_log1p,
            target_bin=target_bin,
            antibiotic_level=antibiotic_level,
            antibiotic_meta_df=antibiotic_meta_df,
            error_on_unmapped=error_on_unmapped,
        )
        embeddings = None

    pathogens = feature_df.index
    amr_weights_df = _prepare_amr_df(amr_df, value_col=value_col)
    pathogen_weights = _compute_sample_weights(
        amr_weights_df,
        pathogens=pathogens,
        value_col=value_col,
        sample_weight_mode=sample_weight_mode,
    )

    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for i in range(n_repeats):
        train_idx, test_idx = train_test_split(np.arange(len(pathogens)), test_size=test_size, random_state=random_state + i)

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        sample_weights = None if pathogen_weights is None else pathogen_weights[train_idx]

        base_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + i,
            n_jobs=-1,
        )
        model = MultiOutputRegressor(base_model)
        if sample_weights is None:
            model.fit(X_train, Y_train)
        else:
            model.fit(X_train, Y_train, sample_weight=sample_weights)

        Y_pred = model.predict(X_test)
        overall_metrics, _ = evaluate_predictions(
            Y_test, Y_pred, antibiotics=_antibiotics_from_target_columns(target_df.columns)
        )
        r2 = overall_metrics["R2"]
        rmse = overall_metrics["RMSE"]
        mae = overall_metrics["MAE"]

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)

        print(f"Repeat {i+1}/{n_repeats}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

    results = pd.DataFrame({"R2": r2_scores, "RMSE": rmse_scores, "MAE": mae_scores})
    avg_scores = {
        "R2_mean": np.mean(r2_scores),
        "R2_std": np.std(r2_scores),
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores),
        "MAE_mean": np.mean(mae_scores),
        "MAE_std": np.std(mae_scores),
    }

    print("\nAverage Performance:")
    print(f"  MAE  = {avg_scores['MAE_mean']:.3f} ± {avg_scores['MAE_std']:.3f}")
    print(f"  RMSE = {avg_scores['RMSE_mean']:.3f} ± {avg_scores['RMSE_std']:.3f}")
    print(f"  R²   = {avg_scores['R2_mean']:.3f} ± {avg_scores['R2_std']:.3f}")

    return results, avg_scores


def rolling_origin_experiment_with_embeddings(
    amr_df: pd.DataFrame,
    data_df: pd.DataFrame,
    embedding_cols=None,
    value_col: str = "count",
    apply_log1p: bool | None = None,
    n_repeats: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth=None,
    antibiotic_level: str = "antibiotic",
    antibiotic_meta_df: pd.DataFrame | None = None,
    error_on_unmapped: bool = True,
    split_strategy: str = "random",
    sample_weight_mode: str = "count_if_available",
):
    """Evaluate across all possible target bins (rolling-origin)."""
    df = _prepare_amr_df(amr_df, value_col=value_col)
    bins = sorted(df["start_date"].unique())
    if len(bins) < 3:
        raise ValueError("Need at least 3 bins for rolling-origin evaluation.")

    strategy = str(split_strategy).strip().lower()
    all_rows = []
    for b in bins[1:]:
        try:
            if strategy == "random":
                res, _ = random_forest_experiment_with_embeddings(
                    amr_df=df,
                    data_df=data_df,
                    embedding_cols=embedding_cols,
                    value_col=value_col,
                    apply_log1p=apply_log1p,
                    n_repeats=n_repeats,
                    test_size=test_size,
                    random_state=random_state,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    target_bin=b,
                    antibiotic_level=antibiotic_level,
                    antibiotic_meta_df=antibiotic_meta_df,
                    error_on_unmapped=error_on_unmapped,
                    sample_weight_mode=sample_weight_mode,
                )
            elif strategy == "lopo":
                lopo = leave_one_pathogen_out_with_embeddings(
                    amr_df=df,
                    data_df=data_df,
                    embedding_cols=embedding_cols,
                    value_col=value_col,
                    apply_log1p=apply_log1p,
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_repeats=n_repeats,
                    antibiotic_level=antibiotic_level,
                    antibiotic_meta_df=antibiotic_meta_df,
                    error_on_unmapped=error_on_unmapped,
                    target_bin=b,
                )
                actual = lopo.pivot(index="pathogen", columns="antibiotic", values="actual_value").sort_index()
                pred = lopo.pivot(index="pathogen", columns="antibiotic", values="predicted_value").reindex(actual.index)
                overall, _ = evaluate_predictions(
                    actual.to_numpy(),
                    pred.to_numpy(),
                    antibiotics=actual.columns.tolist(),
                )
                res = pd.DataFrame([overall])
            else:
                raise ValueError("split_strategy must be 'random' or 'lopo'")
        except ValueError as exc:
            print(f"Skipping target bin {pd.to_datetime(b).date()}: {exc}")
            continue
        res = res.copy()
        res["target_bin"] = pd.to_datetime(b).date().isoformat()
        all_rows.append(res)

    if not all_rows:
        raise ValueError("No rolling-origin splits could be evaluated.")
    results = pd.concat(all_rows, ignore_index=True)
    summary = {
        "R2_mean": float(results["R2"].mean()),
        "R2_std": float(results["R2"].std(ddof=0)),
        "RMSE_mean": float(results["RMSE"].mean()),
        "RMSE_std": float(results["RMSE"].std(ddof=0)),
        "MAE_mean": float(results["MAE"].mean()),
        "MAE_std": float(results["MAE"].std(ddof=0)),
    }
    return results, summary


def baseline_persistence(feature_df: pd.DataFrame, target_df: pd.DataFrame):
    """Baseline: predict next bin as the last observed bin per antibiotic."""
    preds = []
    n_rows = len(feature_df)
    for ab in target_df.columns:
        ab_name = ab.replace("_target", "")
        last_cols = [c for c in feature_df.columns if c.startswith(ab_name + "_")]
        if last_cols:
            last_col = sorted(last_cols)[-1]
            preds.append(feature_df[last_col].values)
        else:
            # No history exists for this antibiotic in feature bins.
            preds.append(np.zeros(n_rows, dtype=float))
    Y_pred = np.vstack(preds).T
    Y_true = target_df.to_numpy()

    r2 = skmetrics.r2_score(Y_true, Y_pred, multioutput="uniform_average")
    rmse = np.sqrt(skmetrics.mean_squared_error(Y_true, Y_pred))
    return r2, rmse


def baseline_global_average(target_df: pd.DataFrame):
    """Baseline: predict global average per antibiotic for all pathogens."""
    Y_true = target_df.to_numpy()
    global_means = Y_true.mean(axis=0, keepdims=True)
    Y_pred = np.repeat(global_means, len(Y_true), axis=0)

    r2 = skmetrics.r2_score(Y_true, Y_pred, multioutput="uniform_average")
    rmse = np.sqrt(skmetrics.mean_squared_error(Y_true, Y_pred))
    return r2, rmse


def leave_one_pathogen_out_with_embeddings(
    amr_df: pd.DataFrame,
    data_df: pd.DataFrame,
    embedding_cols=None,
    value_col: str = "count",
    apply_log1p: bool | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
    n_repeats: int = 3,
    antibiotic_level: str = "antibiotic",
    antibiotic_meta_df: pd.DataFrame | None = None,
    error_on_unmapped: bool = True,
    target_bin=None,
):
    """Leave-One-Pathogen-Out experiment with optional pathogen embeddings."""
    use_log1p = _should_apply_log1p(value_col, apply_log1p)
    if embedding_cols:
        X, Y, feature_df, target_df, embeddings = build_amr_with_embeddings(
            amr_df,
            data_df,
            embedding_cols,
            value_col=value_col,
            apply_log1p=use_log1p,
            target_bin=target_bin,
            antibiotic_level=antibiotic_level,
            antibiotic_meta_df=antibiotic_meta_df,
            error_on_unmapped=error_on_unmapped,
        )
    else:
        X, Y, feature_df, target_df = build_data_and_target(
            amr_df,
            value_col=value_col,
            apply_log1p=use_log1p,
            target_bin=target_bin,
            antibiotic_level=antibiotic_level,
            antibiotic_meta_df=antibiotic_meta_df,
            error_on_unmapped=error_on_unmapped,
        )

    pathogens = feature_df.index
    antibiotics = [col.replace("_target", "") for col in target_df.columns]

    all_results = []

    for i, pathogen in enumerate(pathogens):
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y, i, axis=0)
        X_test = X[i].reshape(1, -1)
        Y_test = Y[i].reshape(1, -1)

        repeat_preds = []
        for r in range(max(1, int(n_repeats))):
            model = MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state + r,
                    n_jobs=-1,
                )
            )
            model.fit(X_train, Y_train)
            Y_pred_r = model.predict(X_test)
            repeat_preds.append(Y_pred_r.flatten())
        repeat_preds_arr = np.vstack(repeat_preds)
        Y_pred_mean = repeat_preds_arr.mean(axis=0)
        Y_pred_std = repeat_preds_arr.std(axis=0, ddof=0)

        if use_log1p:
            actual_raw = np.expm1(Y_test.flatten())
            pred_raw = np.expm1(Y_pred_mean)
            pred_std_raw = np.expm1(repeat_preds_arr).std(axis=0, ddof=0)
        else:
            actual_raw = Y_test.flatten()
            pred_raw = Y_pred_mean
            pred_std_raw = Y_pred_std
        diff = pred_raw - actual_raw

        for ab, actual, pred, pred_std, d in zip(antibiotics, actual_raw, pred_raw, pred_std_raw, diff):
            all_results.append(
                {
                    "pathogen": pathogen,
                    "antibiotic": ab,
                    "actual_value": actual,
                    "predicted_value": pred,
                    "predicted_std": pred_std,
                    "difference": d,
                    "value_col": value_col,
                    "antibiotic_level": antibiotic_level,
                    # Backward-compatible aliases for old notebook cells.
                    "actual_count": actual,
                    "predicted_count": pred,
                }
            )

        print(f"Done pathogen {i+1}/{len(pathogens)}: {pathogen}")

    results_df = pd.DataFrame(all_results)
    return results_df


__all__ = [
    "build_taxonomy_onehot",
    "build_data_and_target",
    "build_amr_with_embeddings",
    "random_forest_experiment_with_embeddings",
    "rolling_origin_experiment_with_embeddings",
    "baseline_persistence",
    "baseline_global_average",
    "evaluate_predictions",
    "leave_one_pathogen_out_with_embeddings",
]
