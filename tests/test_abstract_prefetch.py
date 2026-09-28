import pandas as pd

from amr.screening import core


def test_prefetch_deduplicates_and_reuses_cached_abstracts(tmp_path, monkeypatch):
    cache_path = tmp_path / "abstracts.parquet"
    pd.DataFrame(
        [{"pmid": "10", "title": "cached", "abstract": "already here"}]
    ).to_parquet(cache_path, index=False)
    calls = []

    def fake_fetch(pmids, **kwargs):
        calls.append(list(pmids))
        return {
            "20": ("new title", "new abstract", "ok"),
            "30": ("title only", "", "no_abstract"),
        }

    monkeypatch.setattr(core, "fetch_pubmed_abstracts_batch", fake_fetch)
    candidate_df = pd.DataFrame({"pmid": ["10", "20", "20", "30", None]})

    result = core.prefetch_candidate_abstracts(
        candidate_df, abstract_cache_path=cache_path, batch_size=2
    )

    assert calls == [["20", "30"]]
    assert result.set_index("pmid").loc["10", "abstract"] == "already here"
    assert result.set_index("pmid").loc["20", "abstract"] == "new abstract"
    assert result.set_index("pmid").loc["30", "abstract"] == ""
    assert len(result) == 3


def test_prefetch_retries_failed_fetch_on_resume(tmp_path, monkeypatch):
    cache_path = tmp_path / "abstracts.parquet"
    calls = []

    def fake_fetch(pmids, **kwargs):
        calls.append(list(pmids))
        if len(calls) == 1:
            return {"40": ("", "", "network_error")}
        return {"40": ("recovered", "abstract", "ok")}

    monkeypatch.setattr(core, "fetch_pubmed_abstracts_batch", fake_fetch)
    candidate_df = pd.DataFrame({"pmid": ["40"]})

    first = core.prefetch_candidate_abstracts(
        candidate_df, abstract_cache_path=cache_path
    )
    second = core.prefetch_candidate_abstracts(
        candidate_df, abstract_cache_path=cache_path
    )

    assert calls == [["40"], ["40"]]
    assert first.empty
    assert second.iloc[0]["abstract"] == "abstract"
