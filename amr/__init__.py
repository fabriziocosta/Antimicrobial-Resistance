"""Utilities for antimicrobial resistance analysis notebooks."""

__all__ = [
    "pubmed",
    "pipelines",
    "collect_yearly_amr_dataset",
    "collect_yearly_background_counts",
    "build_yearly_si_dataset",
    "rank_future_high_interest_pairs",
]


def __getattr__(name: str):
    """Load notebook-facing helpers lazily so optional dependencies stay optional."""
    if name == "pubmed":
        from importlib import import_module

        return import_module(".pubmed", __name__)
    if name == "pipelines":
        from importlib import import_module

        return import_module(".pipelines", __name__)
    if name in {"collect_yearly_amr_dataset", "collect_yearly_background_counts"}:
        from .pipelines import collection

        return getattr(collection, name)
    if name in {"build_yearly_si_dataset", "rank_future_high_interest_pairs"}:
        from .pipelines import si

        return getattr(si, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
