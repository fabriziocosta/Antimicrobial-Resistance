"""Predictive modeling helpers for AMR notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


def build_taxonomy_onehot(df: pd.DataFrame, taxonomy_cols=None):
    """Build one-hot embedding vectors for taxonomic information."""
    if taxonomy_cols is None:
        taxonomy_cols = ["domain", "kingdom", "phylum", "class", "order", "family"]

    df_onehot = pd.get_dummies(df[taxonomy_cols], prefix=taxonomy_cols)
    embeddings = df_onehot.values.astype(int).tolist()

    df = df.copy()
    df["taxonomy_embeddings"] = embeddings
    return df, df_onehot


def build_data_and_target(df: pd.DataFrame):
    """Build feature (X) and target (Y) matrices for pathogens."""
    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"])

    all_bins = sorted(df["start_date"].unique())
    last_bin = all_bins[-1]
    feature_bins = all_bins[:-1]

    feature_df = df[df["start_date"].isin(feature_bins)].pivot_table(
        index="pathogen",
        columns=["antibiotic", "start_date"],
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    feature_df = np.log1p(feature_df)
    feature_df.columns = [f"{antibiotic}_{start.date()}" for antibiotic, start in feature_df.columns]

    target_df = df[df["start_date"] == last_bin].pivot_table(
        index="pathogen",
        columns="antibiotic",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    target_df = np.log1p(target_df)
    target_df.columns = [f"{antibiotic}_target" for antibiotic in target_df.columns]

    X = feature_df.to_numpy()
    Y = target_df.to_numpy()

    return X, Y, feature_df, target_df


def build_amr_with_embeddings(amr_df: pd.DataFrame, data_df: pd.DataFrame, embedding_cols):
    """Build AMR features/targets and concatenate pathogen embeddings."""
    X_amr, Y, feature_df, target_df = build_data_and_target(amr_df)

    pathogens = feature_df.index
    data_sub = data_df.set_index("CanonicalSpecies").loc[pathogens]

    embeddings_all = []
    for col in embedding_cols:
        emb = np.vstack(data_sub[col].apply(np.array).values)
        embeddings_all.append(emb)

    embeddings_aligned = np.hstack(embeddings_all)
    X = np.hstack([X_amr, embeddings_aligned])
    return X, Y, feature_df, target_df, embeddings_aligned


def random_forest_experiment_with_embeddings(
    amr_df: pd.DataFrame,
    data_df: pd.DataFrame,
    embedding_cols=None,
    n_repeats: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth=None,
):
    """Run repeated train/test experiments with RandomForest on pathogen AMR features."""
    if embedding_cols:
        X, Y, feature_df, target_df, embeddings = build_amr_with_embeddings(amr_df, data_df, embedding_cols)
    else:
        X, Y, feature_df, target_df = build_data_and_target(amr_df)
        embeddings = None

    pathogens = feature_df.index
    pathogen_totals = amr_df.groupby("pathogen")["count"].sum()
    pathogen_weights = np.log1p(pathogen_totals.loc[pathogens].values)

    r2_scores = []
    rmse_scores = []

    for i in range(n_repeats):
        train_idx, test_idx = train_test_split(np.arange(len(pathogens)), test_size=test_size, random_state=random_state + i)

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        sample_weights = pathogen_weights[train_idx]

        base_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + i,
            n_jobs=-1,
        )
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, Y_train, sample_weight=sample_weights)

        Y_pred = model.predict(X_test)

        r2 = skmetrics.r2_score(Y_test, Y_pred, multioutput="uniform_average")
        rmse = np.sqrt(skmetrics.mean_squared_error(Y_test, Y_pred))

        r2_scores.append(r2)
        rmse_scores.append(rmse)

        print(f"Repeat {i+1}/{n_repeats}: R²={r2:.3f}, RMSE={rmse:.3f}")

    results = pd.DataFrame({"R2": r2_scores, "RMSE": rmse_scores})
    avg_scores = {
        "R2_mean": np.mean(r2_scores),
        "R2_std": np.std(r2_scores),
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores),
    }

    print("\nAverage Performance:")
    print(f"  R²   = {avg_scores['R2_mean']:.3f} ± {avg_scores['R2_std']:.3f}")
    print(f"  RMSE = {avg_scores['RMSE_mean']:.3f} ± {avg_scores['RMSE_std']:.3f}")

    return results, avg_scores


def baseline_persistence(feature_df: pd.DataFrame, target_df: pd.DataFrame):
    """Baseline: predict next bin as the last observed bin per antibiotic."""
    preds = []
    for ab in target_df.columns:
        ab_name = ab.replace("_target", "")
        last_cols = [c for c in feature_df.columns if c.startswith(ab_name + "_")]
        last_col = sorted(last_cols)[-1]
        preds.append(feature_df[last_col].values)
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
    n_estimators: int = 200,
    random_state: int = 42,
):
    """Leave-One-Pathogen-Out experiment with optional pathogen embeddings."""
    if embedding_cols:
        X, Y, feature_df, target_df, embeddings = build_amr_with_embeddings(amr_df, data_df, embedding_cols)
    else:
        X, Y, feature_df, target_df = build_data_and_target(amr_df)

    pathogens = feature_df.index
    antibiotics = [col.replace("_target", "") for col in target_df.columns]

    all_results = []

    for i, pathogen in enumerate(pathogens):
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y, i, axis=0)
        X_test = X[i].reshape(1, -1)
        Y_test = Y[i].reshape(1, -1)

        model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )
        )
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        actual_raw = np.expm1(Y_test.flatten())
        pred_raw = np.expm1(Y_pred.flatten())
        diff = pred_raw - actual_raw

        for ab, actual, pred, d in zip(antibiotics, actual_raw, pred_raw, diff):
            all_results.append(
                {
                    "pathogen": pathogen,
                    "antibiotic": ab,
                    "actual_count": actual,
                    "predicted_count": pred,
                    "difference": d,
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
    "baseline_persistence",
    "baseline_global_average",
    "leave_one_pathogen_out_with_embeddings",
]
