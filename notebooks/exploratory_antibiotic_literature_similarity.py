"""Analysis pipeline for exploratory antibiotic literature similarity."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import datetime as dt
import json
import os
import re
import numpy as np
import pandas as pd
import requests
import yaml
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
try:
    from imste import IteratedMinimumSpanningTreeEmbedder
except ImportError as exc:
    raise RuntimeError("Install the Atlas V2 [viz] extra in the notebook environment before running this analysis.") from exc

@dataclass(frozen=True)
def write_future_si_profile_payload(analysis: dict, output_path: Path) -> Path:
    """Write the precomputed browser payload for the Future-SI explorer plot."""
    output_path = Path(output_path)
    profile_columns = [
        "antibiotic", "imste_1", "imste_2", "n_available_bacteria",
        "effective_sample_size", "largest_normalized_weight",
        "bootstrap_centroid_radius_95", "weighted_feature_dispersion",
    ]
    records = json.loads(analysis["antibiotic_map"][profile_columns].to_json(orient="records"))
    payload = {
        "title": "Future-SI-weighted Atlas profiles",
        "reference_year": int(analysis["selected_horizon"].reference_year),
        "future_year": int(analysis["selected_horizon"].future_year),
        "metric": "cosine",
        "feature_dimensions": len(analysis["normalized_feature_columns"]),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"Future-SI profile plot data: {output_path} ({len(records)} antibiotics)")
    return output_path


class AnalysisSettings:
    project_root: Path
    atlas_root: Path
    ast_data_dir: Path | None = None
    vector_metric: str = "cosine"
    min_species_per_antibiotic: int = 5
    min_available_bacteria: int = 1
    min_shared_species_for_display: int = 1
    ast_min_isolates_per_species_drug: int = 20
    ast_min_species_per_drug: int = 2
    ast_min_total_isolates_per_drug: int = 30
    imste_parameters: dict | None = None
    configured_imste_learning_rate: float | None = None

def run_analysis(settings: AnalysisSettings) -> dict:
    """Load sources and calculate antibiotic profiles, distances, and diagnostics."""
    PROJECT_ROOT = Path(settings.project_root).expanduser().resolve()
    ATLAS_ROOT = Path(settings.atlas_root).expanduser().resolve()
    VECTOR_METRIC = settings.vector_metric
    MIN_SPECIES_PER_ANTIBIOTIC = settings.min_species_per_antibiotic
    MIN_AVAILABLE_BACTERIA = settings.min_available_bacteria
    MIN_SHARED_SPECIES_FOR_DISPLAY = settings.min_shared_species_for_display
    AST_MIN_ISOLATES_PER_SPECIES_DRUG = settings.ast_min_isolates_per_species_drug
    AST_MIN_SPECIES_PER_DRUG = settings.ast_min_species_per_drug
    AST_MIN_TOTAL_ISOLATES_PER_DRUG = settings.ast_min_total_isolates_per_drug
    AMR_PAIRS_PATH = PROJECT_ROOT / "outputs/parquets/yearly_amr_screened_pairs.parquet"
    AMR_PROVENANCE_PATH = AMR_PAIRS_PATH.with_suffix(".provenance.json")
    ATLAS_FEATURES_PATH = ATLAS_ROOT / "data/04_tables/release/species_feature_matrix.parquet"
    ATLAS_FEATURE_SUMMARY_PATH = ATLAS_ROOT / "data/04_tables/release/species_feature_summary.parquet"
    FORECAST_PATH = PROJECT_ROOT / "outputs/parquets/yearly_si_ranked_future.parquet"
    AST_DATA_DIR = Path(settings.ast_data_dir or os.environ.get("AST_COMPARISON_DATA_DIR", PROJECT_ROOT / "outputs/clinical_ast_comparison")).expanduser().resolve()
    for path in [AMR_PAIRS_PATH, ATLAS_FEATURES_PATH, ATLAS_FEATURE_SUMMARY_PATH, FORECAST_PATH]:
        if not path.exists(): raise FileNotFoundError(f"Required input not found: {path}")
    print(f"AMR project: {PROJECT_ROOT}")
    print(f"Atlas V2:    {ATLAS_ROOT}")

    amr_pairs = pd.read_parquet(AMR_PAIRS_PATH)
    atlas_features = pd.read_parquet(ATLAS_FEATURES_PATH)
    atlas_feature_summary = pd.read_parquet(ATLAS_FEATURE_SUMMARY_PATH)
    provenance = json.loads(AMR_PROVENANCE_PATH.read_text()) if AMR_PROVENANCE_PATH.exists() else {}

    required_amr = {"pathogen", "antibiotic", "year", "count"}
    required_atlas = {"canonical_species"}
    assert required_amr.issubset(amr_pairs.columns), f"AMR schema missing {required_amr - set(amr_pairs.columns)}"
    assert required_atlas.issubset(atlas_features.columns)
    assert {"canonical_species", "question_id", "n_answered", "n_yes", "fraction_yes"}.issubset(atlas_feature_summary.columns)

    # Exclude clearly joined combination formulations; hyphenated single-drug/prodrug names are retained.
    COMBINATION_PATTERN = re.compile(r"/|\+|&|\band\b", flags=re.IGNORECASE)
    COMBINATION_EXACT_LABELS = {"sultamicillin"}  # established ampicillin–sulbactam prodrug formulation

    def is_combination_antibiotic(label):
        text = str(label).strip()
        return bool(COMBINATION_PATTERN.search(text) or text.casefold() in COMBINATION_EXACT_LABELS)

    amr_combo_labels = sorted(label for label in amr_pairs.antibiotic.dropna().unique() if is_combination_antibiotic(label))
    amr_pairs = amr_pairs.loc[~amr_pairs.antibiotic.map(is_combination_antibiotic)].copy()
    print(f"Excluded {len(amr_combo_labels)} combination labels from AMR pair analysis: {', '.join(amr_combo_labels)}")

    print("AMR screen provenance:", provenance or "not available")
    print(f"AMR rows: {len(amr_pairs):,}; pair-year range: {amr_pairs.year.min()}–{amr_pairs.year.max()}")
    print(f"Distinct AMR pathogens: {amr_pairs.pathogen.nunique():,}; antibiotics: {amr_pairs.antibiotic.nunique():,}")
    print(f"Atlas species with feature vectors: {atlas_features.canonical_species.nunique():,}")
    def canonical_key(value):
        return re.sub(r"\s+", " ", str(value).strip()).casefold()

    atlas_features = atlas_features.copy()
    atlas_features["species_key"] = atlas_features["canonical_species"].map(canonical_key)
    atlas_feature_summary = atlas_feature_summary.copy()
    atlas_feature_summary["species_key"] = atlas_feature_summary["canonical_species"].map(canonical_key)

    amr_species = pd.DataFrame({"pathogen": sorted(amr_pairs.pathogen.dropna().unique())})
    amr_species["species_key"] = amr_species.pathogen.map(canonical_key)
    feature_columns = [c for c in atlas_features.columns if c.startswith("feature_")]
    feature_lookup = atlas_features.set_index("species_key")[["canonical_species", *feature_columns]]
    matched_species = amr_species.merge(feature_lookup.reset_index(), on="species_key", how="left", validate="one_to_one")
    matched_species["atlas_feature_count"] = matched_species[feature_columns].notna().sum(axis=1)
    matched_species["atlas_feature_coverage"] = matched_species["atlas_feature_count"] / len(feature_columns)

    coverage = pd.DataFrame({
        "metric": ["distinct AMR pathogens", "matched to Atlas feature matrix", "unmatched to Atlas feature matrix", "Atlas feature questions"],
        "value": [len(matched_species), int(matched_species.canonical_species.notna().sum()), int(matched_species.canonical_species.isna().sum()), len(feature_columns)],
    })
    matched_keys = set(matched_species.loc[matched_species.canonical_species.notna(), "species_key"])
    matched_feature_rows = matched_species.loc[matched_species.canonical_species.notna(), ["species_key", "pathogen", *feature_columns]].copy()
    observed_cells = int(matched_feature_rows[feature_columns].notna().sum().sum())
    total_cells = int(len(matched_feature_rows) * len(feature_columns))
    print(f"Observed Atlas feature cells: {observed_cells:,}/{total_cells:,} ({observed_cells / total_cells:.1%})")

    context_question_ids = ["feature_multidrug_resistance", "feature_nosocomial", "feature_bacteraemia", "feature_outbreaks", "feature_urinary", "feature_respiratory"]
    context_columns = [c for c in context_question_ids if c in feature_columns]
    context = matched_feature_rows[context_columns].agg(["count", "mean"]).T.rename(columns={"count": "n_species_scored", "mean": "mean_fraction_yes"})
    question_meta = atlas_feature_summary[atlas_feature_summary.question_id.isin([c.removeprefix("feature_") for c in context_columns])][["question_id", "concept", "category"]].drop_duplicates("question_id")
    context["question_id"] = context.index.str.removeprefix("feature_")
    context = context.reset_index(names="feature").merge(question_meta, on="question_id", how="left")

    # Per-species denominators are kept available for inspection rather than treating missing cells as no.
    answer_coverage = (
        atlas_feature_summary[atlas_feature_summary.species_key.isin(matched_keys)]
        .groupby("canonical_species", as_index=False)
        .agg(n_question_cells=("question_id", "nunique"), median_abstracts_scored=("n_answered", "median"), sum_question_level_answers=("n_answered", "sum"))
        .sort_values("sum_question_level_answers", ascending=False)
    )
    pair_counts = (
        amr_pairs.groupby(["antibiotic", "pathogen"], as_index=False)
        .agg(screened_record_count=("count", "sum"), years_with_records=("year", "nunique"))
    )
    antibiotic_species_counts = (
        pair_counts.groupby("antibiotic", as_index=False)
        .agg(n_species_with_records=("pathogen", "nunique"), n_screened_records=("screened_record_count", "sum"))
        .sort_values(["n_species_with_records", "n_screened_records"], ascending=False)
    )
    footprint_antibiotics = antibiotic_species_counts.loc[
        antibiotic_species_counts.n_species_with_records >= MIN_SPECIES_PER_ANTIBIOTIC, "antibiotic"
    ].tolist()
    footprints = {
        antibiotic: set(pair_counts.loc[pair_counts.antibiotic.eq(antibiotic), "pathogen"])
        for antibiotic in footprint_antibiotics
    }
    print(f"Antibiotics in primary map (at least {MIN_SPECIES_PER_ANTIBIOTIC} species): {len(footprints)} / {amr_pairs.antibiotic.nunique()}")
    antibiotics = footprint_antibiotics
    n = len(antibiotics)
    similarity = np.eye(n, dtype=float)
    shared_support = np.zeros((n, n), dtype=int)
    union_support = np.zeros((n, n), dtype=int)
    for i, a in enumerate(antibiotics):
        for j in range(i, n):
            b = antibiotics[j]
            inter = len(footprints[a] & footprints[b])
            union = len(footprints[a] | footprints[b])
            score = inter / union if union else np.nan
            similarity[i, j] = similarity[j, i] = score
            shared_support[i, j] = shared_support[j, i] = inter
            union_support[i, j] = union_support[j, i] = union

    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)

    # Inspect the closest distinct pairs, with support and both marginal footprint sizes.
    closest_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            closest_pairs.append({
                "antibiotic_a": antibiotics[i], "antibiotic_b": antibiotics[j],
                "jaccard_similarity": similarity[i, j], "shared_species": shared_support[i, j],
                "union_species": union_support[i, j],
                "species_a": len(footprints[antibiotics[i]]), "species_b": len(footprints[antibiotics[j]]),
            })
    closest_pairs = pd.DataFrame(closest_pairs).sort_values(["jaccard_similarity", "shared_species"], ascending=False)

    # Diagonal is 1 by definition. Show support size alongside each antibiotic.
    profile_summary = antibiotic_species_counts[antibiotic_species_counts.antibiotic.isin(antibiotics)].copy()
    # Classical MDS from the distance matrix: eigendecompose the double-centered squared distances.
    D2 = distance ** 2
    row_means = D2.mean(axis=1, keepdims=True)
    column_means = D2.mean(axis=0, keepdims=True)
    grand_mean = D2.mean()
    B = -0.5 * (D2 - row_means - column_means + grand_mean)
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > 1e-10
    if positive.sum() < 2:
        raise ValueError("The selected antibiotic footprints do not support a two-dimensional projection.")
    coords = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    coordinates = pd.DataFrame({"antibiotic": antibiotics, "mds_1": coords[:, 0], "mds_2": coords[:, 1]})
    coordinates = coordinates.merge(profile_summary[["antibiotic", "n_species_with_records", "n_screened_records"]], on="antibiotic", how="left")

    FORECAST_PATH = PROJECT_ROOT / "outputs/parquets/yearly_si_ranked_future.parquet"
    MIN_AVAILABLE_BACTERIA = 1
    forecast_all = pd.read_parquet(FORECAST_PATH)
    required_forecast = {"pathogen", "antibiotic", "status", "reference_year", "future_year", "future_predicted_SI"}
    assert required_forecast.issubset(forecast_all.columns), f"Forecast schema missing {required_forecast - set(forecast_all.columns)}"
    forecast_combo_labels = sorted(label for label in forecast_all.antibiotic.dropna().unique() if is_combination_antibiotic(label))
    forecast_all = forecast_all.loc[~forecast_all.antibiotic.map(is_combination_antibiotic)].copy()
    print(f"Excluded {len(forecast_combo_labels)} combination labels from forecast weighting and map: {', '.join(forecast_combo_labels)}")

    # Keep a single common forecast horizon so the SI values are comparable.
    horizon_counts = (
        forecast_all.loc[forecast_all.status.eq("ok")]
        .groupby(["reference_year", "future_year"], dropna=False)
        .size().rename("n_forecasts").reset_index()
        .sort_values("n_forecasts", ascending=False)
    )
    if horizon_counts.empty:
        raise ValueError("No successful projected-SI forecasts are available.")
    selected_horizon = horizon_counts.iloc[0]
    forecasts = forecast_all.loc[
        forecast_all.status.eq("ok")
        & forecast_all.reference_year.eq(selected_horizon.reference_year)
        & forecast_all.future_year.eq(selected_horizon.future_year)
    ].copy()
    forecasts["future_predicted_SI"] = pd.to_numeric(forecasts.future_predicted_SI, errors="coerce")
    forecasts["species_key"] = forecasts.pathogen.map(canonical_key)
    if forecasts.duplicated(["species_key", "antibiotic"]).any():
        raise ValueError("Forecast table has duplicate bacterium–antibiotic rows at the selected horizon.")

    print(f"Selected forecast horizon: {int(selected_horizon.reference_year)} → {int(selected_horizon.future_year)}")
    print(f"Successful forecasts at that horizon: {len(forecasts):,} across {forecasts.pathogen.nunique()} bacteria and {forecasts.antibiotic.nunique()} antibiotics")
    import yaml
    from sklearn.preprocessing import StandardScaler
    try:
        from imste import IteratedMinimumSpanningTreeEmbedder
    except ImportError as exc:
        raise RuntimeError(
            "The Atlas V2 visualization extra provides IMSTE. Install the repository's [viz] extra in the notebook environment before running this embedding cell; this notebook does not install dependencies."
        ) from exc

    raw_atlas_features = atlas_features.copy()
    raw_feature_columns = [column for column in raw_atlas_features.columns if column.startswith("feature_")]
    numeric_atlas = raw_atlas_features[raw_feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric_atlas = numeric_atlas.dropna(axis=1, how="all")
    mean_imputed_atlas = numeric_atlas.fillna(numeric_atlas.mean(axis=0))
    feature_variances = mean_imputed_atlas.var(axis=0, ddof=0)
    normalized_feature_columns = feature_variances[feature_variances > 0].index.tolist()
    if len(normalized_feature_columns) < 2:
        raise ValueError("Atlas V2 needs at least two nonconstant scored features for IMSTE.")
    scaler = StandardScaler()
    normalized_atlas_values = scaler.fit_transform(mean_imputed_atlas[normalized_feature_columns].to_numpy(dtype=float))
    normalized_atlas = pd.DataFrame(normalized_atlas_values, columns=normalized_feature_columns, index=raw_atlas_features.species_key)
    if normalized_atlas.index.duplicated().any():
        raise ValueError("Atlas feature matrix has duplicate canonical species keys.")

    print(f"Atlas species used to fit normalization: {len(normalized_atlas):,}")
    print(f"Atlas feature dimensions retained after removing constants: {len(normalized_feature_columns)}")
    print("Missing Atlas cells: imputed using the full Atlas species mean before standardization, matching pathogen_atlas_v2/plots/embeddings.py")
    forecast_pairs = forecasts.merge(
        normalized_atlas.reset_index(names="species_key"), on="species_key", how="left", validate="many_to_one", suffixes=("", "_atlas")
    )
    forecast_pairs["atlas_vector_present"] = forecast_pairs[normalized_feature_columns].notna().all(axis=1)
    forecast_pairs["finite_forecast"] = forecast_pairs.future_predicted_SI.notna()

    # Per-antibiotic coverage includes failed rows and is displayed alongside the map output.
    all_horizon = forecast_all.loc[
        forecast_all.reference_year.eq(selected_horizon.reference_year)
        & forecast_all.future_year.eq(selected_horizon.future_year)
    ].copy()
    all_horizon["species_key"] = all_horizon.pathogen.map(canonical_key)
    all_horizon = all_horizon.merge(
        normalized_atlas.reset_index(names="species_key"), on="species_key", how="left", validate="many_to_one", suffixes=("", "_atlas")
    )
    all_horizon["atlas_vector_present"] = all_horizon[normalized_feature_columns].notna().all(axis=1)
    all_horizon["finite_forecast"] = pd.to_numeric(all_horizon.future_predicted_SI, errors="coerce").notna()
    all_horizon["eligible_pair"] = all_horizon.status.eq("ok") & all_horizon.finite_forecast & all_horizon.atlas_vector_present
    coverage_summary = (
        all_horizon.groupby("antibiotic", as_index=False)
        .agg(
            n_forecast_rows=("pathogen", "size"),
            n_successful_forecasts=("status", lambda s: int(s.eq("ok").sum())),
            n_finite_forecasts=("finite_forecast", "sum"),
            n_matched_finite_pairs=("eligible_pair", "sum"),
            n_forecast_bacteria=("species_key", "nunique"),
        )
    )
    matched_bacteria_counts = (
        all_horizon.loc[all_horizon.atlas_vector_present]
        .groupby("antibiotic").species_key.nunique()
        .rename("n_matched_bacteria").reset_index()
    )
    coverage_summary = coverage_summary.merge(matched_bacteria_counts, on="antibiotic", how="left")
    coverage_summary["n_matched_bacteria"] = coverage_summary.n_matched_bacteria.fillna(0).astype(int)

    forecast_weight_diagnostics = pd.DataFrame([
        {"diagnostic": "successful pair forecasts at selected horizon", "n_pairs": len(forecast_pairs), "n_bacteria": forecast_pairs.species_key.nunique(), "n_antibiotics": forecast_pairs.antibiotic.nunique()},
        {"diagnostic": "finite projected SI values", "n_pairs": int(forecast_pairs.finite_forecast.sum()), "n_bacteria": forecast_pairs.loc[forecast_pairs.finite_forecast, "species_key"].nunique(), "n_antibiotics": forecast_pairs.loc[forecast_pairs.finite_forecast, "antibiotic"].nunique()},
        {"diagnostic": "missing projected SI values", "n_pairs": int((~forecast_pairs.finite_forecast).sum()), "n_bacteria": forecast_pairs.loc[~forecast_pairs.finite_forecast, "species_key"].nunique(), "n_antibiotics": forecast_pairs.loc[~forecast_pairs.finite_forecast, "antibiotic"].nunique()},
        {"diagnostic": "rows without an Atlas vector", "n_pairs": int((~forecast_pairs.atlas_vector_present).sum()), "n_bacteria": forecast_pairs.loc[~forecast_pairs.atlas_vector_present, "species_key"].nunique(), "n_antibiotics": forecast_pairs.loc[~forecast_pairs.atlas_vector_present, "antibiotic"].nunique()},
        {"diagnostic": "finite forecast rows with an Atlas vector", "n_pairs": int((forecast_pairs.finite_forecast & forecast_pairs.atlas_vector_present).sum()), "n_bacteria": forecast_pairs.loc[forecast_pairs.finite_forecast & forecast_pairs.atlas_vector_present, "species_key"].nunique(), "n_antibiotics": forecast_pairs.loc[forecast_pairs.finite_forecast & forecast_pairs.atlas_vector_present, "antibiotic"].nunique()},
    ])
    if forecast_pairs.finite_forecast.any():
        print("Future SI range among finite successful forecasts:", forecast_pairs.future_predicted_SI.min(), "to", forecast_pairs.future_predicted_SI.max())

    eligible_pairs = forecast_pairs.loc[forecast_pairs.finite_forecast & forecast_pairs.atlas_vector_present].copy()
    # Stable softmax-equivalent weights: exp(SI - max(SI)) / sum(exp(SI - max(SI))).
    eligible_pairs["max_si_within_antibiotic"] = eligible_pairs.groupby("antibiotic").future_predicted_SI.transform("max")
    eligible_pairs["relative_exp_weight"] = np.exp(eligible_pairs.future_predicted_SI - eligible_pairs.max_si_within_antibiotic)
    relative_weight_sums = eligible_pairs.groupby("antibiotic").relative_exp_weight.transform("sum")
    eligible_pairs["normalized_weight"] = eligible_pairs.relative_exp_weight / relative_weight_sums
    eligible_pairs["n_available_bacteria"] = eligible_pairs.groupby("antibiotic").species_key.transform("nunique")

    weight_summary = (
        eligible_pairs.groupby("antibiotic", as_index=False)
        .agg(
            n_available_bacteria=("species_key", "nunique"),
            min_future_SI=("future_predicted_SI", "min"),
            max_future_SI=("future_predicted_SI", "max"),
            normalized_weight_sum=("normalized_weight", "sum"),
            largest_normalized_weight=("normalized_weight", "max"),
            effective_sample_size=("normalized_weight", lambda w: 1.0 / np.square(w).sum()),
        )
    )

    # Weighted bacterial-profile dispersion is descriptive heterogeneity. A species bootstrap radius summarizes
    # how much the weighted centroid moves when the observed bacteria are resampled from their normalized weights.
    rng = np.random.default_rng(42)
    N_BOOTSTRAP = 500
    profile_diagnostics = []
    for antibiotic, subset in eligible_pairs.groupby("antibiotic", sort=False):
        weights = subset.normalized_weight.to_numpy(dtype=float)
        species_vectors = normalized_atlas.loc[subset.species_key, normalized_feature_columns].to_numpy(dtype=float)
        centroid = np.average(species_vectors, axis=0, weights=weights)
        within_profile_dispersion = float(np.sqrt(np.sum(weights * np.square(np.linalg.norm(species_vectors - centroid, axis=1)))))
        bootstrap_radii = np.empty(N_BOOTSTRAP, dtype=float)
        for draw in range(N_BOOTSTRAP):
            sampled_indices = rng.choice(len(weights), size=len(weights), replace=True, p=weights)
            bootstrap_centroid = species_vectors[sampled_indices].mean(axis=0)
            bootstrap_radii[draw] = np.linalg.norm(bootstrap_centroid - centroid)
        profile_diagnostics.append({
            "antibiotic": antibiotic,
            "weighted_feature_dispersion": within_profile_dispersion,
            "bootstrap_centroid_radius_95": float(np.quantile(bootstrap_radii, 0.95)),
        })
    profile_diagnostics = pd.DataFrame(profile_diagnostics)
    weight_summary = weight_summary.merge(profile_diagnostics, on="antibiotic", how="left", validate="one_to_one")
    weight_summary["included_in_imste"] = weight_summary.n_available_bacteria >= MIN_AVAILABLE_BACTERIA
    weight_summary["support_flag"] = np.where(weight_summary.n_available_bacteria == 1, "single bacterium", "multi-bacterium")
    weight_summary = weight_summary.merge(coverage_summary, on="antibiotic", how="left", validate="one_to_one")
    weight_summary["map_status"] = np.where(weight_summary.included_in_imste, "included", "omitted: no matched finite forecast")
    coverage_summary = coverage_summary.merge(weight_summary[["antibiotic", "effective_sample_size", "largest_normalized_weight", "bootstrap_centroid_radius_95", "weighted_feature_dispersion", "map_status"]], on="antibiotic", how="left", validate="one_to_one")
    included_antibiotics = weight_summary.loc[weight_summary.included_in_imste, "antibiotic"].tolist()
    if len(included_antibiotics) < 3:
        raise ValueError("Fewer than three antibiotics have finite forecast weights and matched Atlas features.")
    weighted_vectors = []
    for antibiotic in included_antibiotics:
        subset = eligible_pairs.loc[eligible_pairs.antibiotic.eq(antibiotic)]
        weights = subset.normalized_weight.to_numpy(dtype=float)
        vector_matrix = normalized_atlas.loc[subset.species_key, normalized_feature_columns].to_numpy(dtype=float)
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"Normalized weights for {antibiotic} do not sum to 1.")
        weighted_vectors.append(np.average(vector_matrix, axis=0, weights=weights))
    antibiotic_vectors = pd.DataFrame(weighted_vectors, index=included_antibiotics, columns=normalized_feature_columns)
    antibiotic_vectors.index.name = "antibiotic"
    print(f"Exponentially weighted antibiotic vectors: {antibiotic_vectors.shape[0]} antibiotics × {antibiotic_vectors.shape[1]} normalized Atlas features")
    atlas_settings = (
        dict(settings.imste_parameters)
        if settings.imste_parameters is not None
        else yaml.safe_load((ATLAS_ROOT / "configs/notebook_parameters.yaml").read_text())["phase5_plot_creation"]
    )
    n_points = len(antibiotic_vectors)
    CONFIGURED_IMSTE_LEARNING_RATE = float(
        settings.configured_imste_learning_rate
        if settings.configured_imste_learning_rate is not None
        else atlas_settings["imste_learning_rate"]
    )
    IMSTE_LEARNING_RATE = float(atlas_settings["imste_learning_rate"])
    IMSTE_GRAVITY_MODE = atlas_settings["gravity_mode"]
    print(
        f"IMSTE settings: metric={VECTOR_METRIC}; n_epochs={int(atlas_settings['imste_n_epochs'])}; "
        f"configured learning_rate={CONFIGURED_IMSTE_LEARNING_RATE}; used learning_rate={IMSTE_LEARNING_RATE}; "
        f"gravity={IMSTE_GRAVITY_MODE}"
    )
    mst_neighbors = min(int(atlas_settings["imste_mst_neighbors"]), n_points - 1)
    mst_max_value = atlas_settings.get("imste_mst_max_neighbors")
    mst_max_neighbors = None if mst_max_value is None else max(mst_neighbors, min(int(mst_max_value), n_points - 1))

    embedder = IteratedMinimumSpanningTreeEmbedder(
        n_msts=int(atlas_settings["imste_n_msts"]),
        n_epochs=int(atlas_settings["imste_n_epochs"]),
        mst_method=str(atlas_settings["imste_mst_method"]),
        mst_neighbors=mst_neighbors,
        mst_inter_component_edges=int(atlas_settings["imste_mst_inter_component_edges"]),
        mst_max_neighbors=mst_max_neighbors,
        negative_ratio=int(atlas_settings["imste_negative_ratio"]),
        lambda_rep=float(atlas_settings["imste_lambda_rep"]),
        learning_rate=IMSTE_LEARNING_RATE,
        batch_size=int(atlas_settings["imste_batch_size"]),
        epsilon=float(atlas_settings["imste_epsilon"]),
        logistic_margin=float(atlas_settings["imste_logistic_margin"]),
        logistic_temperature=float(atlas_settings["imste_logistic_temperature"]),
        mst_representatives_per_component=int(atlas_settings["imste_mst_representatives_per_component"]),
        random_state=int(atlas_settings["imste_random_state"]),
        device=str(atlas_settings["imste_device"]),
        metric=VECTOR_METRIC,
        gravity_mode=atlas_settings["gravity_mode"],
        gravity_neighbors=int(atlas_settings["gravity_neighbors"]),
        gravity_resolution=float(atlas_settings["gravity_resolution"]),
        gravity_strength=float(atlas_settings["gravity_strength"]),
        gravity_core_quantile=float(atlas_settings["gravity_core_quantile"]),
        gravity_mass_power=float(atlas_settings["gravity_mass_power"]),
        gravity_attraction_strength=float(atlas_settings["gravity_attraction_strength"]),
        gravity_near_repulsion_strength=float(atlas_settings["gravity_near_repulsion_strength"]),
        gravity_force_neighbors=int(atlas_settings["gravity_force_neighbors"]),
        gravity_attraction_range=float(atlas_settings["gravity_attraction_range"]),
        gravity_repulsion_range=float(atlas_settings["gravity_repulsion_range"]),
        gravity_step_size=float(atlas_settings["gravity_step_size"]),
        gravity_iterations=int(atlas_settings["gravity_iterations"]),
        gravity_pack_regions=bool(atlas_settings["gravity_pack_regions"]),
        gravity_pack_radius_quantile=float(atlas_settings["gravity_pack_radius_quantile"]),
        gravity_pack_gap_fraction=float(atlas_settings["gravity_pack_gap_fraction"]),
        mean_shift_neighbors=int(atlas_settings["mean_shift_neighbors"]),
        mean_shift_bandwidth_factor=float(atlas_settings["mean_shift_bandwidth_factor"]),
        mean_shift_step_size=float(atlas_settings["mean_shift_step_size"]),
        mean_shift_iterations=int(atlas_settings["mean_shift_iterations"]),
    )
    imste_coordinates = embedder.fit_transform(antibiotic_vectors.to_numpy(dtype=float))
    if imste_coordinates.shape != (n_points, 2):
        raise ValueError(f"Expected two-dimensional IMSTE coordinates, got {imste_coordinates.shape}.")
    antibiotic_map = pd.DataFrame({"antibiotic": included_antibiotics, "imste_1": imste_coordinates[:, 0], "imste_2": imste_coordinates[:, 1]})
    antibiotic_map = antibiotic_map.merge(weight_summary, on="antibiotic", how="left", validate="one_to_one")
    # Compare the high-dimensional profiles with their distances in the 2D IMSTE map.
    from itertools import combinations
    from scipy.spatial.distance import pdist, squareform

    # Cosine distance is exactly 1 - cosine similarity in the shared 73D feature space.
    cosine_distance_matrix = squareform(pdist(antibiotic_vectors.to_numpy(dtype=float), metric=VECTOR_METRIC))
    map_coordinates = antibiotic_map.set_index("antibiotic")[["imste_1", "imste_2"]]
    profile_rows = []
    for antibiotic_a, antibiotic_b in combinations(antibiotic_vectors.index, 2):
        vector_a = antibiotic_vectors.loc[antibiotic_a].to_numpy(dtype=float)
        vector_b = antibiotic_vectors.loc[antibiotic_b].to_numpy(dtype=float)
        high_d_cosine_distance = float(cosine_distance_matrix[antibiotic_vectors.index.get_loc(antibiotic_a), antibiotic_vectors.index.get_loc(antibiotic_b)])
        cosine = 1.0 - high_d_cosine_distance
        xy_a = map_coordinates.loc[antibiotic_a].to_numpy(dtype=float)
        xy_b = map_coordinates.loc[antibiotic_b].to_numpy(dtype=float)
        support_a = eligible_pairs.loc[eligible_pairs.antibiotic.eq(antibiotic_a)].set_index("pathogen").normalized_weight
        support_b = eligible_pairs.loc[eligible_pairs.antibiotic.eq(antibiotic_b)].set_index("pathogen").normalized_weight
        shared_species = set(support_a.index) & set(support_b.index)
        shared_weight_mass = float(sum(min(support_a[s], support_b[s]) for s in shared_species))
        profile_rows.append({
            "antibiotic_a": antibiotic_a,
            "antibiotic_b": antibiotic_b,
            "cosine_distance_73d": high_d_cosine_distance,
            "cosine_similarity_73d": cosine,
            "imste_distance_2d": float(np.linalg.norm(xy_a - xy_b)),
            "shared_species_count": len(shared_species),
            "shared_weight_mass": shared_weight_mass,
        })
    profile_pair_diagnostics = pd.DataFrame(profile_rows)
    non_tied_cosine_pairs = profile_pair_diagnostics.loc[profile_pair_diagnostics.cosine_distance_73d.gt(1e-8)]
    profile_distance_rank_correlation = non_tied_cosine_pairs.cosine_distance_73d.corr(non_tied_cosine_pairs.imste_distance_2d, method="spearman")
    print(f"Spearman rank correlation between 73D cosine distances and 2D distances (excluding exact cosine ties): {profile_distance_rank_correlation:.3f}")

    INTERPRETATION_PAIRS = [
        ("Ceftazidime", "Meropenem"),
        ("Aztreonam", "Cefepime"),
        ("Teicoplanin", "Vancomycin"),
        ("Amikacin", "Ceftazidime"),
    ]
    interpretation_table = profile_pair_diagnostics.set_index(["antibiotic_a", "antibiotic_b"]).loc[INTERPRETATION_PAIRS].reset_index()
    for antibiotic in set(sum((list(pair) for pair in INTERPRETATION_PAIRS), [])):
        summary = weight_summary.set_index("antibiotic").loc[antibiotic]
        interpretation_table.loc[
            interpretation_table.antibiotic_a.eq(antibiotic), "support_a"
        ] = f"n={int(summary.n_available_bacteria)}, ESS={summary.effective_sample_size:.2f}, boot95={summary.bootstrap_centroid_radius_95:.2f}"
        interpretation_table.loc[
            interpretation_table.antibiotic_b.eq(antibiotic), "support_b"
        ] = f"n={int(summary.n_available_bacteria)}, ESS={summary.effective_sample_size:.2f}, boot95={summary.bootstrap_centroid_radius_95:.2f}"
    import datetime as dt
    import requests
    from scipy.stats import spearmanr, pearsonr

    AST_DATA_DIR = Path(os.environ.get("AST_COMPARISON_DATA_DIR", PROJECT_ROOT / "outputs/clinical_ast_comparison")).expanduser().resolve()
    AST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    AST_SOURCE_PATH = AST_DATA_DIR / "ncbi_pathogen_detection_clinical_ast_atlas_forecast_extract.parquet"
    AST_SOURCE_CSV_PATH = AST_SOURCE_PATH.with_suffix(".csv")
    AST_SUMMARY_PATH = AST_DATA_DIR / "species_antibiotic_clinical_ast_summary.csv"
    AST_COVERAGE_PATH = AST_DATA_DIR / "clinical_ast_coverage_by_antibiotic.csv"
    AST_MANIFEST_PATH = AST_DATA_DIR / "manifest.json"
    AST_MIN_ISOLATES_PER_SPECIES_DRUG = 20
    AST_MIN_SPECIES_PER_DRUG = 2
    AST_MIN_TOTAL_ISOLATES_PER_DRUG = 30
    AST_SOURCE_URL = "https://www.ncbi.nlm.nih.gov/pathogens/ast/"
    AST_ENDPOINT = "https://www.ncbi.nlm.nih.gov/pathogens/pathogens-srv/"
    AST_FIELDS = "id,biosample_acc,taxgroup_name,scientific_name,epi_type,geo_loc_name,target_acc,isolation_source,host,collection_date,antibiotic,phenotype,measurement_sign,mic,disk_diffusion,standard,platform,vendor,reagent,bioproject_acc"

    # Reuse the local filtered source extract. On a fresh checkout, retrieve the current
    # clinical-human AST rows once; the endpoint returns a compact tabular record set.
    if AST_SOURCE_PATH.exists():
        ast_download = pd.read_parquet(AST_SOURCE_PATH)
        ast_download_mode = "loaded saved filtered source extract"
    else:
        ast_session = requests.Session()
        ast_base_params = {
            "action": "retrieve", "collection": "ast",
            "fq": 'epi_type:clinical AND host:"Homo sapiens"',
            "fl": AST_FIELDS,
        }
        first = ast_session.post(AST_ENDPOINT, data={**ast_base_params, "start": 0, "limit": 1}, timeout=90)
        first.raise_for_status()
        ast_total = int(first.json()["ngout"]["data"]["totalCount"])
        ast_batches = []
        AST_PAGE_SIZE = 5000
        for ast_start in range(0, ast_total, AST_PAGE_SIZE):
            response = ast_session.post(
                AST_ENDPOINT, data={**ast_base_params, "start": ast_start, "limit": AST_PAGE_SIZE}, timeout=120
            )
            response.raise_for_status()
            batch = response.json().get("ngout", {}).get("data", {}).get("content", [])
            if not batch:
                raise RuntimeError(f"NCBI AST download returned an empty page at offset {ast_start}.")
            ast_batches.extend(batch)
        ast_download = pd.DataFrame(ast_batches)
        ast_download_mode = f"downloaded {len(ast_download):,} current clinical-human AST rows from NCBI"

    # Only exact species/antibiotic pairs in the successful forecast horizon are eligible.
    def normalize_antibiotic_key(value):
        value = re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", str(value).strip()))
        value = re.sub(r"[-_]+", " ", value).casefold()
        value = re.sub(r"\s+", " ", value).strip()
        return {"rifampicin": "rifampin"}.get(value, value)

    forecast_ast_pairs = forecasts[["pathogen", "antibiotic", "species_key", "future_predicted_SI"]].copy().rename(columns={"antibiotic": "forecast_antibiotic"})
    forecast_ast_pairs["antibiotic_key"] = forecast_ast_pairs.forecast_antibiotic.map(normalize_antibiotic_key)
    forecast_ast_pairs = forecast_ast_pairs.drop_duplicates(["species_key", "antibiotic_key"])
    feature_species = atlas_features[["species_key", "canonical_species"]].drop_duplicates("species_key")
    ast_raw_n = len(ast_download)
    ast = ast_download.copy()
    ast["phenotype_norm"] = ast.phenotype.fillna("").astype(str).str.strip().str.casefold()
    ast["standard_norm"] = ast.standard.fillna("").astype(str).str.strip().str.casefold()
    ast = ast.loc[
        ast.epi_type.fillna("").astype(str).str.casefold().eq("clinical")
        & ast.host.fillna("").astype(str).str.casefold().eq("homo sapiens")
        & ast.standard_norm.eq("clsi")
        & ast.phenotype_norm.isin(["susceptible", "intermediate", "resistant"])
    ].copy()
    ast = ast.drop(columns=["pathogen", "forecast_antibiotic", "future_predicted_SI"], errors="ignore")
    ast["species_key"] = ast.scientific_name.map(canonical_key)
    ast["antibiotic_key"] = ast.antibiotic.map(normalize_antibiotic_key)
    ast = ast.merge(forecast_ast_pairs, on=["species_key", "antibiotic_key"], how="inner", validate="many_to_one")
    ast = ast.drop(columns=["canonical_species"], errors="ignore").merge(feature_species, on="species_key", how="inner", validate="many_to_one")
    ast_source_extract = ast.copy()

    # Save the individual filtered phenotype rows before deduplication for traceability.
    source_export_columns = [
        "id", "biosample_acc", "target_acc", "scientific_name", "canonical_species", "taxgroup_name",
        "epi_type", "host", "geo_loc_name", "isolation_source", "collection_date", "antibiotic",
        "standard", "phenotype", "measurement_sign", "mic", "disk_diffusion", "platform", "vendor",
        "reagent", "bioproject_acc", "pathogen", "future_predicted_SI",
    ]
    source_export_columns = list(dict.fromkeys(c for c in source_export_columns if c in ast_source_extract.columns))
    ast_source_extract[source_export_columns].drop_duplicates().to_parquet(AST_SOURCE_PATH, index=False)
    ast_source_extract[source_export_columns].drop_duplicates().to_csv(AST_SOURCE_CSV_PATH, index=False)

    ast["isolate_id"] = ast.target_acc.fillna(ast.biosample_acc)
    if ast.isolate_id.isna().any():
        raise ValueError("Some NCBI AST rows have neither a target nor BioSample accession.")
    repeat_key = ["isolate_id", "species_key", "antibiotic_key"]
    phenotype_n = ast.groupby(repeat_key).phenotype_norm.nunique()
    conflicting_keys = set(phenotype_n[phenotype_n.gt(1)].index)
    repeat_index = pd.MultiIndex.from_frame(ast[repeat_key])
    conflicting_mask = repeat_index.isin(conflicting_keys)
    conflicting_row_count = int(conflicting_mask.sum())
    ast_dedup = ast.loc[~conflicting_mask].drop_duplicates(repeat_key).copy()
    ast_summary = (
        ast_dedup.groupby(["canonical_species", "species_key", "pathogen", "antibiotic", "antibiotic_key"], as_index=False)
        .agg(
            n_susceptible=("phenotype_norm", lambda s: int(s.eq("susceptible").sum())),
            n_intermediate=("phenotype_norm", lambda s: int(s.eq("intermediate").sum())),
            n_resistant=("phenotype_norm", lambda s: int(s.eq("resistant").sum())),
            n_isolates=("isolate_id", "nunique"),
        )
    )
    ast_summary["n_interpretable_AST"] = ast_summary[["n_susceptible", "n_intermediate", "n_resistant"]].sum(axis=1)
    ast_summary["susceptible_fraction"] = ast_summary.n_susceptible / ast_summary.n_interpretable_AST
    ast_summary["intermediate_fraction"] = ast_summary.n_intermediate / ast_summary.n_interpretable_AST
    ast_summary["resistant_fraction"] = ast_summary.n_resistant / ast_summary.n_interpretable_AST
    # Wilson 95% interval for the isolate-level susceptible fraction, shown per species-drug pair.
    z95 = 1.959963984540054
    n_ast = ast_summary.n_interpretable_AST.to_numpy(dtype=float)
    p_s = ast_summary.susceptible_fraction.to_numpy(dtype=float)
    wilson_denom = 1 + z95**2 / n_ast
    wilson_center = (p_s + z95**2 / (2 * n_ast)) / wilson_denom
    wilson_half = z95 * np.sqrt(p_s * (1-p_s) / n_ast + z95**2 / (4*n_ast**2)) / wilson_denom
    ast_summary["susceptible_ci95_low"] = wilson_center - wilson_half
    ast_summary["susceptible_ci95_high"] = wilson_center + wilson_half
    ast_summary["standard"] = "CLSI"
    ast_summary["isolation_type"] = "clinical"
    ast_summary["host"] = "Homo sapiens"
    ast_summary["unit"] = "unique target accession × species × antibiotic; conflicting repeats excluded"
    ast_summary.to_csv(AST_SUMMARY_PATH, index=False)

    # Explicitly record all forecast pairs, including missing AST observations.
    FORECAST_AST_PAIR_COVERAGE_PATH = AST_DATA_DIR / "forecast_ast_pair_coverage.csv"
    forecast_pair_coverage = (
        forecast_ast_pairs[["pathogen", "forecast_antibiotic", "species_key", "antibiotic_key", "future_predicted_SI"]]
        .rename(columns={"pathogen": "forecast_pathogen"})
        .merge(
            ast_summary[["species_key", "antibiotic_key", "n_isolates", "n_interpretable_AST", "susceptible_fraction"]],
            on=["species_key", "antibiotic_key"], how="left", validate="one_to_one"
        )
    )
    forecast_pair_coverage["has_clinical_CLSI_SIR"] = forecast_pair_coverage.n_interpretable_AST.notna()
    forecast_pair_coverage["meets_20_isolates"] = forecast_pair_coverage.n_interpretable_AST.ge(AST_MIN_ISOLATES_PER_SPECIES_DRUG)
    forecast_pair_coverage.to_csv(FORECAST_AST_PAIR_COVERAGE_PATH, index=False)

    ast_support = ast_summary.loc[ast_summary.n_interpretable_AST.ge(AST_MIN_ISOLATES_PER_SPECIES_DRUG)].copy()
    ast_coverage = (
        ast_summary.groupby("antibiotic", as_index=False)
        .agg(
            n_species_with_AST=("canonical_species", "nunique"),
            n_isolates=("n_isolates", "sum"),
            n_interpretable_AST=("n_interpretable_AST", "sum"),
        )
    )
    threshold_counts = ast_support.groupby("antibiotic").canonical_species.nunique().rename("n_species_ge20").reset_index()
    ast_coverage = ast_coverage.merge(threshold_counts, on="antibiotic", how="left")
    ast_coverage["n_species_ge20"] = ast_coverage.n_species_ge20.fillna(0).astype(int)
    ast_coverage["eligible_for_comparison"] = ast_coverage.n_species_ge20.ge(AST_MIN_SPECIES_PER_DRUG) & ast_coverage.n_isolates.ge(AST_MIN_TOTAL_ISOLATES_PER_DRUG)
    ast_coverage.to_csv(AST_COVERAGE_PATH, index=False)

    # Keep source and transformation provenance beside the local extract.
    manifest = json.loads(AST_MANIFEST_PATH.read_text()) if AST_MANIFEST_PATH.exists() else {}
    manifest.update({
        "source": "NCBI Pathogen Detection AST Browser / BioSample submitter-provided phenotypic AST",
        "source_url": AST_SOURCE_URL,
        "source_documentation_url": "https://www.ncbi.nlm.nih.gov/pathogens/docs/ast/",
        "ncbi_reuse_policy_url": "https://www.ncbi.nlm.nih.gov/home/about/policies/",
        "biosample_submission_faq_url": "https://www.ncbi.nlm.nih.gov/biosample/docs/submission/faq/",
        "download_endpoint": AST_ENDPOINT + "?action=retrieve&collection=ast",
        "last_notebook_run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notebook_load_mode": ast_download_mode,
        "notebook_source_rows_loaded": ast_raw_n,
        "filtered_extract_rows": int(len(ast_source_extract)),
        "filtered_unique_species_drug_cells": int(len(ast_summary)),
        "filtered_species": int(ast_summary.canonical_species.nunique()),
        "filtered_antibiotics": int(ast_summary.antibiotic.nunique()),
        "selected_horizon_forecast_pairs": int(len(forecast_pair_coverage)),
        "selected_horizon_pairs_with_AST": int(forecast_pair_coverage.has_clinical_CLSI_SIR.sum()),
        "comparison_antibiotics": int(ast_coverage.eligible_for_comparison.sum()),
        "filters_applied": ["clinical isolation type", "host Homo sapiens", "CLSI standard", "phenotype in S/I/R", "exact Atlas V2 species", "exact successful future-SI pair after name normalization"],
        "deduplication": "unique target accession × species × antibiotic; duplicate calls collapsed; conflicting phenotype groups excluded",
        "conflicting_isolate_drug_groups_excluded": int(len(conflicting_keys)),
        "conflicting_source_rows_excluded": int(conflicting_row_count),
        "minimum_interpretable_isolates_per_species_drug": int(AST_MIN_ISOLATES_PER_SPECIES_DRUG),
        "minimum_species_per_antibiotic": int(AST_MIN_SPECIES_PER_DRUG),
        "minimum_total_isolates_per_antibiotic": int(AST_MIN_TOTAL_ISOLATES_PER_DRUG),
        "license_note": "NCBI's general policies distinguish US-government-created pages from third-party submitted records. The AST browser does not attach a separate license to returned phenotype rows; NCBI says submitters remain responsible for their content and may retain rights. Keep accession attribution and check source/submitter terms before redistribution.",
    })
    AST_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"NCBI data: {ast_download_mode}; source rows currently loaded={ast_raw_n:,}")
    print(f"Filtered exact forecast/Atlas pairs: {len(ast_source_extract):,} rows, {ast_summary.shape[0]:,} species-drug cells, {ast_summary.canonical_species.nunique()} species, {ast_summary.antibiotic.nunique()} drugs")
    print(f"Dropped {len(conflicting_keys)} conflicting isolate-drug groups ({conflicting_row_count} AST rows); retained only CLSI S/I/R calls.")
    print(f"Forecast pair overlap: {int(forecast_pair_coverage.has_clinical_CLSI_SIR.sum())}/{len(forecast_pair_coverage)} successful pairs have at least one clinical CLSI S/I/R call.")
    print(f"AST comparison support: {int(ast_coverage.eligible_for_comparison.sum())} antibiotics meet >=2 species with >=20 AST each and >=30 isolates total.")
    # Restrict both sets of weights to the same species-drug cells with sufficient AST support.
    common_pairs = ast_support.merge(
        forecast_ast_pairs[["species_key", "antibiotic_key", "future_predicted_SI", "forecast_antibiotic"]],
        on=["species_key", "antibiotic_key"], how="inner", validate="one_to_one", suffixes=("_ast", "_forecast")
    )
    # Canonical forecast label is authoritative for matching, display, and the SI map.
    if "forecast_antibiotic" not in common_pairs:
        common_pairs = common_pairs.rename(columns={"antibiotic": "forecast_antibiotic"})
    common_pairs["max_si"] = common_pairs.groupby("forecast_antibiotic").future_predicted_SI.transform("max")
    common_pairs["si_relative_weight"] = np.exp(common_pairs.future_predicted_SI - common_pairs.max_si)
    common_pairs["si_weight"] = common_pairs.si_relative_weight / common_pairs.groupby("forecast_antibiotic").si_relative_weight.transform("sum")
    common_pairs["susceptible_mass"] = common_pairs.susceptible_fraction
    common_pairs["susceptible_mass_sum"] = common_pairs.groupby("forecast_antibiotic").susceptible_mass.transform("sum")
    common_pairs = common_pairs.loc[common_pairs.susceptible_mass_sum.gt(0)].copy()
    common_pairs["ast_weight"] = common_pairs.susceptible_mass / common_pairs.susceptible_mass_sum
    comparison_coverage = (
        common_pairs.groupby("forecast_antibiotic", as_index=False)
        .agg(
            n_common_species=("canonical_species", "nunique"),
            n_total_isolates=("n_isolates", "sum"),
            min_species_n=("n_isolates", "min"),
            max_species_n=("n_isolates", "max"),
            effective_sample_size=("ast_weight", lambda w: float(1 / np.square(w).sum())),
            max_ast_weight=("ast_weight", "max"),
            total_susceptible_mass=("susceptible_mass", "sum"),
        )
    )
    comparison_coverage = comparison_coverage.loc[comparison_coverage.n_common_species.ge(AST_MIN_SPECIES_PER_DRUG) & comparison_coverage.n_total_isolates.ge(AST_MIN_TOTAL_ISOLATES_PER_DRUG)].copy()
    comparison_antibiotics = comparison_coverage.forecast_antibiotic.tolist()
    common_pairs = common_pairs.loc[common_pairs.forecast_antibiotic.isin(comparison_antibiotics)].copy()
    comparison_coverage["n_species_drug_cells"] = comparison_coverage.n_common_species
    comparison_coverage.to_csv(AST_DATA_DIR / "comparison_pair_support.csv", index=False)
    common_pairs.to_csv(AST_DATA_DIR / "comparison_species_drug_weights.csv", index=False)

    if len(comparison_antibiotics) < 3:
        raise ValueError("Fewer than three antibiotics have adequate shared clinical AST support for a map comparison.")

    def weighted_centroid_table(weight_name):
        rows=[]
        for abx in comparison_antibiotics:
            subset=common_pairs.loc[common_pairs.forecast_antibiotic.eq(abx)]
            vectors=normalized_atlas.loc[subset.species_key, normalized_feature_columns].to_numpy(dtype=float)
            weights=subset[weight_name].to_numpy(dtype=float)
            if not np.isclose(weights.sum(),1.0):
                raise ValueError(f"{weight_name} values do not sum to one for {abx}")
            rows.append(np.average(vectors,axis=0,weights=weights))
        out=pd.DataFrame(rows,index=comparison_antibiotics,columns=normalized_feature_columns)
        out.index.name="antibiotic"
        return out

    si_common_vectors=weighted_centroid_table("si_weight")
    ast_common_vectors=weighted_centroid_table("ast_weight")

    # Isolate bootstrap: resample S/I/R outcomes within each shared species-drug stratum,
    # recalculate susceptible-fraction weights, then record 73D centroid displacement.
    AST_BOOTSTRAPS=500
    rng_ast=np.random.default_rng(20261114)
    ast_bootstrap_radius=[]
    for abx in comparison_antibiotics:
        subset=common_pairs.loc[common_pairs.forecast_antibiotic.eq(abx)].reset_index(drop=True)
        vectors=normalized_atlas.loc[subset.species_key,normalized_feature_columns].to_numpy(dtype=float)
        empirical_weights=subset.ast_weight.to_numpy(dtype=float)
        centroid=np.average(vectors,axis=0,weights=empirical_weights)
        labels=np.array(["susceptible","intermediate","resistant"])
        cell_probs=subset[["n_susceptible","n_intermediate","n_resistant"]].to_numpy(dtype=float)
        cell_probs=cell_probs/cell_probs.sum(axis=1,keepdims=True)
        boot_centroids=[]
        for _ in range(AST_BOOTSTRAPS):
            boot_s=[]
            for row,n_iso in zip(cell_probs,subset.n_interpretable_AST):
                boot_s.append(rng_ast.multinomial(int(n_iso),row)[0]/int(n_iso))
            boot_w=np.asarray(boot_s,dtype=float)
            if boot_w.sum()>0:
                boot_centroids.append(np.average(vectors,axis=0,weights=boot_w))
        radii=np.linalg.norm(np.asarray(boot_centroids)-centroid,axis=1) if boot_centroids else np.array([np.nan])
        ast_bootstrap_radius.append({"antibiotic":abx,"ast_centroid_radius_95_73d":float(np.nanquantile(radii,.95)),"bootstrap_draws_kept":len(radii)})
    ast_bootstrap_radius=pd.DataFrame(ast_bootstrap_radius)
    comparison_coverage=comparison_coverage.merge(ast_bootstrap_radius,left_on="forecast_antibiotic",right_on="antibiotic",how="left").drop(columns="antibiotic")
    comparison_coverage.to_csv(AST_DATA_DIR / "comparison_pair_support.csv",index=False)

    # Two separately fitted IMSTE maps. Coordinates have arbitrary rotation/reflection;
    # distance concordance below is computed in shared 73D feature space instead.
    comparison_settings=atlas_settings
    comparison_n=len(comparison_antibiotics)
    comparison_mst_neighbors=min(int(comparison_settings["imste_mst_neighbors"]),comparison_n-1)
    comparison_mst_max_value=comparison_settings.get("imste_mst_max_neighbors")
    comparison_mst_max_neighbors=None if comparison_mst_max_value is None else max(comparison_mst_neighbors,min(int(comparison_mst_max_value),comparison_n-1))
    def fit_comparison_imste(vectors):
        embedder=IteratedMinimumSpanningTreeEmbedder(
            n_msts=int(comparison_settings["imste_n_msts"]),n_epochs=int(comparison_settings["imste_n_epochs"]),
            mst_method=str(comparison_settings["imste_mst_method"]),mst_neighbors=comparison_mst_neighbors,
            mst_inter_component_edges=int(comparison_settings["imste_mst_inter_component_edges"]),mst_max_neighbors=comparison_mst_max_neighbors,
            negative_ratio=int(comparison_settings["imste_negative_ratio"]),lambda_rep=float(comparison_settings["imste_lambda_rep"]),
            learning_rate=IMSTE_LEARNING_RATE,batch_size=int(comparison_settings["imste_batch_size"]),
            epsilon=float(comparison_settings["imste_epsilon"]),logistic_margin=float(comparison_settings["imste_logistic_margin"]),
            logistic_temperature=float(comparison_settings["imste_logistic_temperature"]),
            mst_representatives_per_component=int(comparison_settings["imste_mst_representatives_per_component"]),
            random_state=int(comparison_settings["imste_random_state"]),device=str(comparison_settings["imste_device"]),
            metric=VECTOR_METRIC,gravity_mode=comparison_settings["gravity_mode"],
            gravity_neighbors=int(comparison_settings["gravity_neighbors"]),
            gravity_resolution=float(comparison_settings["gravity_resolution"]),
            gravity_strength=float(comparison_settings["gravity_strength"]),
            gravity_core_quantile=float(comparison_settings["gravity_core_quantile"]),
            gravity_mass_power=float(comparison_settings["gravity_mass_power"]),
            gravity_attraction_strength=float(comparison_settings["gravity_attraction_strength"]),
            gravity_near_repulsion_strength=float(comparison_settings["gravity_near_repulsion_strength"]),
            gravity_force_neighbors=int(comparison_settings["gravity_force_neighbors"]),
            gravity_attraction_range=float(comparison_settings["gravity_attraction_range"]),
            gravity_repulsion_range=float(comparison_settings["gravity_repulsion_range"]),
            gravity_step_size=float(comparison_settings["gravity_step_size"]),
            gravity_iterations=int(comparison_settings["gravity_iterations"]),
            gravity_pack_regions=bool(comparison_settings["gravity_pack_regions"]),
            gravity_pack_radius_quantile=float(comparison_settings["gravity_pack_radius_quantile"]),
            gravity_pack_gap_fraction=float(comparison_settings["gravity_pack_gap_fraction"]),
            mean_shift_neighbors=int(comparison_settings["mean_shift_neighbors"]),
            mean_shift_bandwidth_factor=float(comparison_settings["mean_shift_bandwidth_factor"]),
            mean_shift_step_size=float(comparison_settings["mean_shift_step_size"]),
            mean_shift_iterations=int(comparison_settings["mean_shift_iterations"]),
        )
        return embedder.fit_transform(vectors.to_numpy(dtype=float))
    si_xy=fit_comparison_imste(si_common_vectors)
    ast_xy=fit_comparison_imste(ast_common_vectors)
    si_map_common=pd.DataFrame({"antibiotic":comparison_antibiotics,"x":si_xy[:,0],"y":si_xy[:,1]})
    ast_map_common=pd.DataFrame({"antibiotic":comparison_antibiotics,"x":ast_xy[:,0],"y":ast_xy[:,1]}).merge(comparison_coverage,left_on="antibiotic",right_on="forecast_antibiotic",how="left")

    # Per-antibiotic cosine distance agreement in the original shared 73D feature space.
    from scipy.spatial.distance import pdist, squareform
    si_dist=squareform(pdist(si_common_vectors.to_numpy(dtype=float),metric=VECTOR_METRIC))
    ast_dist=squareform(pdist(ast_common_vectors.to_numpy(dtype=float),metric=VECTOR_METRIC))
    tri=np.triu_indices_from(si_dist,k=1)
    si_flat=si_dist[tri];ast_flat=ast_dist[tri]
    rank_corr=float(spearmanr(si_flat,ast_flat).statistic)
    linear_corr=float(pearsonr(si_flat,ast_flat).statistic)
    # Use explicit pair labels in upper-triangle order.
    from itertools import combinations
    pair_distance_comparison=pd.DataFrame([
        {"antibiotic_a":a,"antibiotic_b":b,"si_cosine_distance_73d":float(si_dist[i,j]),"ast_cosine_distance_73d":float(ast_dist[i,j]),"absolute_distance_difference":float(abs(si_dist[i,j]-ast_dist[i,j]))}
        for (i,a),(j,b) in combinations(list(enumerate(comparison_antibiotics)),2)
    ])
    pair_distance_comparison.to_csv(AST_DATA_DIR / "si_vs_ast_pairwise_cosine_distances.csv",index=False)
    print(f"Same-basis map: {comparison_n} antibiotics, {len(common_pairs)} species-drug cells, {len(normalized_feature_columns)} Atlas features; min pair n={int(common_pairs.n_interpretable_AST.min())}, max pair n={int(common_pairs.n_interpretable_AST.max())}.")
    print(f"73D cosine-distance concordance across {len(pair_distance_comparison)} antibiotic pairs: Spearman rho={rank_corr:.3f}; Pearson r={linear_corr:.3f}.")
    # Full pair table reports the exact values behind the scatter plot.
    result_names = [
        "PROJECT_ROOT", "ATLAS_ROOT", "AST_DATA_DIR", "provenance", "amr_pairs", "atlas_features",
        "atlas_feature_summary", "coverage", "matched_species", "context", "answer_coverage",
        "antibiotic_species_counts", "footprint_antibiotics", "footprints", "similarity", "distance",
        "shared_support", "union_support", "closest_pairs", "profile_summary", "coordinates",
        "forecast_all", "forecasts", "selected_horizon", "coverage_summary", "weight_summary",
        "eligible_pairs", "normalized_atlas", "normalized_feature_columns", "antibiotic_vectors",
        "included_antibiotics", "antibiotic_map", "profile_pair_diagnostics", "profile_distance_rank_correlation",
        "ast_coverage", "forecast_pair_coverage", "ast_summary", "comparison_coverage", "common_pairs",
        "comparison_antibiotics", "comparison_n", "si_common_vectors", "ast_common_vectors", "si_map_common", "ast_map_common",
        "pair_distance_comparison", "rank_corr", "linear_corr", "AST_HTML_PATH", "AST_PNG_PATH",
        "ast_download_mode", "ast_raw_n", "forecast_weight_diagnostics", "matched_feature_rows",
        "feature_columns", "atlas_settings", "profile_rows", "INTERPRETATION_PAIRS",
    ]
    local_values = locals()
    return {name: local_values[name] for name in result_names if name in local_values}
