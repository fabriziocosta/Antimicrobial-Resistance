"""Pipeline entry points for yearly collection and scientific-interest workflows."""

__all__ = [
    "count_pmids_per_year_for_antibiotics",
    "count_pmids_per_year_for_antibiotics_resumable",
    "count_pmids_per_year_for_pathogens",
    "count_pmids_per_year_for_pathogens_resumable",
    "fetch_pmids_by_date_and_keywords",
    "fetch_pmids_intervals",
    "build_and_print_collection_preflight",
    "build_collection_preflight_summary",
    "collect_yearly_amr_dataset",
    "collect_yearly_background_counts",
    "coarsen_category_counts",
    "build_yearly_si_dataset",
    "DEFAULT_DECREASING_SI_SUMMARY_COLUMNS",
    "DEFAULT_DRIVER_CATEGORY_COARSENING_PATH",
    "DEFAULT_DRIVER_SUMMARY_COLUMNS",
    "DEFAULT_INTERESTING_SUMMARY_COLUMNS",
    "DEFAULT_SMOOTHED_SI_SHAPE_SUMMARY_COLUMNS",
    "filter_model_inputs_by_average_amr_publications",
    "rank_future_high_interest_pairs",
    "build_si_analysis_paths",
    "build_glass_match_table",
    "build_novel_prediction_table",
    "build_si_driver_analysis",
    "load_or_build_interactive_rank",
    "load_si_and_clean_glass_list",
    "load_yearly_si_inputs",
    "plot_future_si_vs_si_shift",
    "plot_latest_si_vs_future_si",
    "print_collection_preflight",
    "print_si_dataset_summary",
    "print_pubmed_query_examples",
    "print_unified_prompt_examples",
    "print_wrapped_system_prompt",
    "prepare_antibiotic_analysis_level",
    "save_category_driver_plots",
    "save_interesting_si_plots",
    "save_smoothed_si_screen_plots",
    "select_decreasing_smoothed_si_trends",
    "select_existing_columns",
    "select_smoothed_si_shape_cases",
    "slugify_analysis_level",
    "summarize_smoothed_si_shapes",
    "summarize_pair_publication_rates",
    "summarize_pathogen_coverage",
]

_PUBMED_INTERVAL_EXPORTS = {
    "count_pmids_per_year_for_antibiotics",
    "count_pmids_per_year_for_antibiotics_resumable",
    "count_pmids_per_year_for_pathogens",
    "count_pmids_per_year_for_pathogens_resumable",
    "fetch_pmids_by_date_and_keywords",
    "fetch_pmids_intervals",
}
_COLLECTION_EXPORTS = {
    "build_collection_preflight_summary",
    "collect_yearly_amr_dataset",
    "collect_yearly_background_counts",
}
_COLLECTION_PREVIEW_EXPORTS = {
    "build_and_print_collection_preflight",
    "print_collection_preflight",
    "print_pubmed_query_examples",
    "print_unified_prompt_examples",
    "print_wrapped_system_prompt",
}
_SI_EXPORTS = {
    "build_yearly_si_dataset",
    "coarsen_category_counts",
    "DEFAULT_DECREASING_SI_SUMMARY_COLUMNS",
    "DEFAULT_DRIVER_CATEGORY_COARSENING_PATH",
    "DEFAULT_DRIVER_SUMMARY_COLUMNS",
    "DEFAULT_INTERESTING_SUMMARY_COLUMNS",
    "DEFAULT_SMOOTHED_SI_SHAPE_SUMMARY_COLUMNS",
    "filter_model_inputs_by_average_amr_publications",
    "rank_future_high_interest_pairs",
    "build_si_analysis_paths",
    "build_si_driver_analysis",
    "load_yearly_si_inputs",
    "prepare_antibiotic_analysis_level",
    "print_si_dataset_summary",
    "save_category_driver_plots",
    "save_interesting_si_plots",
    "save_smoothed_si_screen_plots",
    "select_decreasing_smoothed_si_trends",
    "select_existing_columns",
    "select_smoothed_si_shape_cases",
    "slugify_analysis_level",
    "summarize_smoothed_si_shapes",
    "summarize_pair_publication_rates",
    "summarize_pathogen_coverage",
}
_GLASS_SI_EXPORTS = {
    "build_glass_match_table",
    "build_novel_prediction_table",
    "load_or_build_interactive_rank",
    "load_si_and_clean_glass_list",
    "plot_future_si_vs_si_shift",
    "plot_latest_si_vs_future_si",
}


def __getattr__(name: str):
    from importlib import import_module

    if name in _PUBMED_INTERVAL_EXPORTS:
        intervals = import_module("amr.pubmed.intervals")
        return getattr(intervals, name)
    if name in _COLLECTION_EXPORTS:
        collection = import_module(".collection", __name__)
        return getattr(collection, name)
    if name in _COLLECTION_PREVIEW_EXPORTS:
        collection_preview = import_module(".collection_preview", __name__)
        return getattr(collection_preview, name)
    if name in _SI_EXPORTS:
        si = import_module(".si", __name__)
        return getattr(si, name)
    if name in _GLASS_SI_EXPORTS:
        glass_si = import_module(".glass_si", __name__)
        return getattr(glass_si, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
