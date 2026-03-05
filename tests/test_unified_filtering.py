import unittest
from unittest.mock import patch

import pandas as pd

from amr.filtering import (
    build_yearly_screened_pairs,
    prepare_yearly_candidate_pmids,
    run_unified_llm_filter,
)


class _FakeLLM:
    def complete(self, prompt: str, **kwargs: object) -> str:
        if "PMID: 111" in prompt:
            return '{"decision":"YES","rationale":"clear resistance in clinical isolates"}'
        if "PMID: 222" in prompt:
            return '{"decision":"No","rationale":"lab model only"}'
        return '{"decision":"uncertain","rationale":"ambiguous"}'


class UnifiedFilteringTests(unittest.TestCase):
    def test_prepare_yearly_candidate_pmids(self) -> None:
        yearly_df = pd.DataFrame(
            {
                "pathogen": ["p1", "p1"],
                "antibiotic": ["a1", "a1"],
                "year": [2020, 2020],
                "pmids": [["111", "222"], ["222", "333"]],
            }
        )
        out = prepare_yearly_candidate_pmids(yearly_df)
        self.assertEqual(list(out.columns), ["pmid", "pathogen", "antibiotic", "year"])
        self.assertEqual(len(out), 3)
        self.assertEqual(set(out["pmid"]), {"111", "222", "333"})

    @patch("amr.filtering.fetch_pubmed_abstract")
    def test_run_unified_llm_filter_normalizes_decisions(self, mock_fetch: patch) -> None:
        mock_fetch.side_effect = [
            ("t1", "clinical resistant isolates", "ok"),
            ("t2", "lab model assay", "ok"),
            ("t3", "", "no_abstract"),
        ]
        candidates = pd.DataFrame(
            {
                "pmid": ["111", "222", "333"],
                "pathogen": ["p1", "p1", "p1"],
                "antibiotic": ["a1", "a1", "a1"],
                "year": [2020, 2020, 2020],
            }
        )

        out = run_unified_llm_filter(
            candidates,
            _FakeLLM(),
            dry_run=False,
            verbose=0,
            abstract_cache_path=None,
            progress_every=0,
        )
        out = out.sort_values("pmid").reset_index(drop=True)
        self.assertEqual(out.loc[0, "unified_decision"], "yes")
        self.assertEqual(out.loc[1, "unified_decision"], "no")
        self.assertEqual(out.loc[2, "unified_decision"], "uncertain")

    def test_build_yearly_screened_pairs_unique_and_counted(self) -> None:
        yearly_df = pd.DataFrame(
            {
                "pathogen": ["p1", "p1"],
                "antibiotic": ["a1", "a1"],
                "year": [2020, 2020],
                "pmids": [["111", "222"], ["222", "333"]],
            }
        )
        unified_df = pd.DataFrame(
            {
                "pmid": ["111", "222", "333"],
                "pathogen": ["p1", "p1", "p1"],
                "antibiotic": ["a1", "a1", "a1"],
                "year": [2020, 2020, 2020],
                "unified_decision": ["yes", "uncertain", "no"],
            }
        )
        out = build_yearly_screened_pairs(yearly_df, unified_df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "count"], 2)
        self.assertEqual(out.loc[0, "pmids"], ["111", "222"])


if __name__ == "__main__":
    unittest.main()
