import unittest
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pandas as pd

from amr.pubmed.yearly import explode_positive_pair_pmids
from amr.screening import (
    AMR_CATEGORY_COLUMNS,
    build_yearly_screened_pairs,
    prepare_yearly_candidate_pmids,
    run_unified_llm_filter,
)


class _FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        if "PMID: 111" in prompt:
            return (
                '{"decision":"YES","primary_category":"SURVEILLANCE",'
                '"rationale":"clear resistance in clinical isolates",'
                '"evidence_phrases":["clinical resistant isolates"],'
                '"target_pair_relation_supported":true,"is_amr_relevant":true}'
            )
        if "PMID: 222" in prompt:
            return (
                '{"decision":"No","primary_category":"NONE",'
                '"rationale":"lab model only","evidence_phrases":["lab model assay"],'
                '"target_pair_relation_supported":false,"is_amr_relevant":false}'
            )
        if "PMID: 444" in prompt:
            return "not json"
        return '{"decision":"uncertain","rationale":"ambiguous"}'


class UnifiedFilteringTests(unittest.TestCase):
    def _complete_amr_fields(self) -> dict:
        return {
            "amr_primary_category": ["SURVEILLANCE"],
            "amr_evidence_phrases": [["clinical resistant isolates"]],
            "amr_target_pair_relation_supported": [True],
            "amr_is_relevant": [True],
        }

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

    def test_explode_positive_pair_pmids_drops_nan_like_pmids(self) -> None:
        source = pd.DataFrame(
            {
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "start_date": ["2020/01/01"],
                "end_date": ["2024/12/31"],
                "pmids": [["111", float("nan"), "nan", None, "PMID:222"]],
            }
        )

        out = explode_positive_pair_pmids(source)

        self.assertEqual(out["pmid"].tolist(), ["111", "222"])

    @patch("amr.screening.core.fetch_pubmed_abstracts_batch")
    @patch("amr.screening.core.fetch_pubmed_abstract")
    def test_run_unified_llm_filter_normalizes_decisions(self, mock_fetch_single: patch, mock_fetch_batch: patch) -> None:
        mock_fetch_batch.return_value = {
            "111": ("t1", "clinical resistant isolates", "ok"),
            "222": ("t2", "lab model assay", "ok"),
            "333": ("t3", "", "no_abstract"),
        }
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
            failed_retry_path=None,
            progress_every=0,
        )
        out = out.sort_values("pmid").reset_index(drop=True)
        self.assertEqual(out.loc[0, "unified_decision"], "yes")
        self.assertEqual(out.loc[1, "unified_decision"], "no")
        self.assertEqual(out.loc[2, "unified_decision"], "uncertain")
        self.assertEqual(out.loc[0, "amr_primary_category"], "SURVEILLANCE")
        self.assertEqual(out.loc[0, "amr_evidence_phrases"], ["clinical resistant isolates"])
        self.assertTrue(out.loc[0, "amr_target_pair_relation_supported"])
        self.assertTrue(out.loc[0, "amr_is_relevant"])
        self.assertEqual(out.loc[2, "amr_primary_category"], "NONE")
        self.assertEqual(out.loc[2, "amr_evidence_phrases"], [])
        self.assertFalse(out.loc[2, "amr_target_pair_relation_supported"])
        self.assertFalse(out.loc[2, "amr_is_relevant"])
        self.assertEqual(mock_fetch_batch.call_count, 1)
        self.assertEqual(mock_fetch_single.call_count, 0)

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

    def test_build_yearly_screened_pairs_merges_with_existing_output(self) -> None:
        yearly_df = pd.DataFrame(
            {
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2021],
                "pmids": [["444"]],
            }
        )
        unified_df = pd.DataFrame(
            {
                "pmid": ["444"],
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2021],
                "unified_decision": ["yes"],
            }
        )
        existing_out = pd.DataFrame(
            {
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2020],
                "count": [1],
                "pmids": [["111"]],
            }
        )
        with NamedTemporaryFile(suffix=".parquet") as tmp:
            existing_out.to_parquet(tmp.name, index=False)
            out = build_yearly_screened_pairs(
                yearly_df,
                unified_df,
                output_path=tmp.name,
                merge_with_existing_output=True,
            )
            self.assertEqual(set(out["year"]), {2020, 2021})

    @patch("amr.screening.core.fetch_pubmed_abstracts_batch")
    @patch("amr.screening.core.fetch_pubmed_abstract")
    def test_run_unified_llm_filter_auto_resume_from_output(self, mock_fetch_single: patch, mock_fetch_batch: patch) -> None:
        mock_fetch_batch.return_value = {"222": ("t", "clinical resistant isolates", "ok")}
        candidates = pd.DataFrame(
            {
                "pmid": ["111", "222"],
                "pathogen": ["p1", "p1"],
                "antibiotic": ["a1", "a1"],
                "year": [2020, 2020],
            }
        )
        existing = pd.DataFrame(
            {
                "pmid": ["111"],
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2020],
                "unified_decision": ["yes"],
                "unified_raw_decision": ["yes"],
                "unified_rationale": ["cached"],
                "title": [""],
                "abstract": [""],
                **self._complete_amr_fields(),
            }
        )
        with NamedTemporaryFile(suffix=".parquet") as tmp:
            existing.to_parquet(tmp.name, index=False)
            out = run_unified_llm_filter(
                candidates,
                _FakeLLM(),
                output_path=tmp.name,
                auto_resume_from_output=True,
                verbose=0,
                abstract_cache_path=None,
                failed_retry_path=None,
                progress_every=0,
            )
            self.assertEqual(len(out), 2)
            # only PMID 222 should require a new abstract fetch
            self.assertEqual(mock_fetch_batch.call_count, 1)
            self.assertEqual(mock_fetch_batch.call_args.args[0], ["222"])
            self.assertEqual(mock_fetch_single.call_count, 0)

    @patch("amr.screening.core.fetch_pubmed_abstracts_batch")
    @patch("amr.screening.core.fetch_pubmed_abstract")
    def test_run_unified_llm_filter_reruns_cached_llm_failures(
        self, mock_fetch_single: patch, mock_fetch_batch: patch
    ) -> None:
        mock_fetch_batch.return_value = {"111": ("t", "clinical resistant isolates", "ok")}
        candidates = pd.DataFrame(
            {
                "pmid": ["111"],
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2020],
            }
        )
        existing = pd.DataFrame(
            {
                "pmid": ["111"],
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2020],
                "unified_decision": ["uncertain"],
                "unified_raw_decision": ["uncertain"],
                "unified_rationale": ["LLM failure: OpenAI completion failed: Error code: 429"],
                "title": ["old title"],
                "abstract": ["old abstract"],
                "amr_primary_category": ["NONE"],
                "amr_evidence_phrases": [[]],
                "amr_target_pair_relation_supported": [False],
                "amr_is_relevant": [False],
            }
        )
        with NamedTemporaryFile(suffix=".parquet") as tmp:
            existing.to_parquet(tmp.name, index=False)
            out = run_unified_llm_filter(
                candidates,
                _FakeLLM(),
                output_path=tmp.name,
                auto_resume_from_output=True,
                verbose=0,
                abstract_cache_path=None,
                failed_retry_path=None,
                progress_every=0,
            )

        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "unified_decision"], "yes")
        self.assertEqual(out.loc[0, "unified_rationale"], "clear resistance in clinical isolates")
        self.assertEqual(out.loc[0, "amr_primary_category"], "SURVEILLANCE")
        self.assertEqual(mock_fetch_batch.call_args.args[0], ["111"])
        self.assertEqual(mock_fetch_single.call_count, 0)

    @patch("amr.screening.core.fetch_pubmed_abstracts_batch")
    @patch("amr.screening.core.fetch_pubmed_abstract")
    def test_run_unified_llm_filter_progress_reports_pending_rows(
        self, mock_fetch_single: patch, mock_fetch_batch: patch
    ) -> None:
        mock_fetch_batch.return_value = {"222": ("t", "clinical resistant isolates", "ok")}
        candidates = pd.DataFrame(
            {
                "pmid": ["111", "222"],
                "pathogen": ["p1", "p1"],
                "antibiotic": ["a1", "a1"],
                "year": [2020, 2020],
            }
        )
        existing = pd.DataFrame(
            {
                "pmid": ["111"],
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2020],
                "unified_decision": ["yes"],
                "unified_raw_decision": ["yes"],
                "unified_rationale": ["cached"],
                "title": [""],
                "abstract": [""],
                **self._complete_amr_fields(),
            }
        )
        with NamedTemporaryFile(suffix=".parquet") as tmp:
            existing.to_parquet(tmp.name, index=False)
            with patch("builtins.print") as mock_print:
                run_unified_llm_filter(
                    candidates,
                    _FakeLLM(),
                    output_path=tmp.name,
                    auto_resume_from_output=True,
                    verbose=1,
                    abstract_cache_path=None,
                    failed_retry_path=None,
                    progress_every=1,
                )

        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("[unified_filter] starting run for 1 pending candidate rows", printed)
        self.assertIn("2 input rows; skipped 1 already complete", printed)
        self.assertIn("Processed      1/1", printed)
        self.assertEqual(mock_fetch_single.call_count, 0)

    @patch("amr.screening.core.fetch_pubmed_abstracts_batch")
    @patch("amr.screening.core.fetch_pubmed_abstract")
    def test_run_unified_llm_filter_backfills_missing_category_fields(
        self, mock_fetch_single: patch, mock_fetch_batch: patch
    ) -> None:
        mock_fetch_batch.return_value = {"111": ("t", "clinical resistant isolates", "ok")}
        candidates = pd.DataFrame(
            {
                "pmid": ["111"],
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2020],
            }
        )
        existing = pd.DataFrame(
            {
                "pmid": ["111"],
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2020],
                "unified_decision": ["yes"],
                "unified_raw_decision": ["yes"],
                "unified_rationale": ["cached without category fields"],
                "title": [""],
                "abstract": [""],
            }
        )
        with NamedTemporaryFile(suffix=".parquet") as tmp:
            existing.to_parquet(tmp.name, index=False)
            out = run_unified_llm_filter(
                candidates,
                _FakeLLM(),
                output_path=tmp.name,
                auto_resume_from_output=True,
                verbose=0,
                abstract_cache_path=None,
                failed_retry_path=None,
                progress_every=0,
            )
            self.assertEqual(len(out), 1)
            self.assertEqual(out.loc[0, "unified_rationale"], "clear resistance in clinical isolates")
            for col in AMR_CATEGORY_COLUMNS:
                self.assertIn(col, out.columns)
            self.assertEqual(out.loc[0, "amr_primary_category"], "SURVEILLANCE")
            self.assertEqual(mock_fetch_batch.call_args.args[0], ["111"])
            self.assertEqual(mock_fetch_single.call_count, 0)

    @patch("amr.screening.core.fetch_pubmed_abstracts_batch")
    @patch("amr.screening.core.fetch_pubmed_abstract")
    def test_run_unified_llm_filter_can_defer_backfill_for_decisions(
        self, mock_fetch_single: patch, mock_fetch_batch: patch
    ) -> None:
        candidates = pd.DataFrame(
            {
                "pmid": ["222"],
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2020],
            }
        )
        existing = pd.DataFrame(
            {
                "pmid": ["222"],
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2020],
                "unified_decision": ["no"],
                "unified_raw_decision": ["no"],
                "unified_rationale": ["cached no without category fields"],
                "title": [""],
                "abstract": [""],
            }
        )
        with NamedTemporaryFile(suffix=".parquet") as tmp:
            existing.to_parquet(tmp.name, index=False)
            out = run_unified_llm_filter(
                candidates,
                _FakeLLM(),
                output_path=tmp.name,
                auto_resume_from_output=True,
                verbose=0,
                abstract_cache_path=None,
                failed_retry_path=None,
                progress_every=0,
                category_backfill_decisions=("yes", "uncertain"),
            )
            self.assertEqual(len(out), 1)
            self.assertEqual(out.loc[0, "unified_decision"], "no")
            self.assertTrue(pd.isna(out.loc[0, "amr_primary_category"]))
            self.assertEqual(mock_fetch_batch.call_count, 0)
            self.assertEqual(mock_fetch_single.call_count, 0)

    @patch("amr.screening.core.fetch_pubmed_abstracts_batch")
    @patch("amr.screening.core.fetch_pubmed_abstract")
    def test_run_unified_llm_filter_malformed_json_gets_safe_defaults(
        self, mock_fetch_single: patch, mock_fetch_batch: patch
    ) -> None:
        mock_fetch_batch.return_value = {"444": ("t", "clinical resistant isolates", "ok")}
        candidates = pd.DataFrame(
            {
                "pmid": ["444"],
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2020],
            }
        )
        out = run_unified_llm_filter(
            candidates,
            _FakeLLM(),
            verbose=0,
            abstract_cache_path=None,
            failed_retry_path=None,
            progress_every=0,
        )
        self.assertEqual(out.loc[0, "unified_decision"], "uncertain")
        self.assertEqual(out.loc[0, "amr_primary_category"], "NONE")
        self.assertEqual(out.loc[0, "amr_evidence_phrases"], [])
        self.assertFalse(out.loc[0, "amr_target_pair_relation_supported"])
        self.assertFalse(out.loc[0, "amr_is_relevant"])
        self.assertEqual(mock_fetch_single.call_count, 0)

    @patch("amr.screening.core.fetch_pubmed_abstracts_batch")
    @patch("amr.screening.core.fetch_pubmed_abstract")
    def test_run_unified_llm_filter_no_abstract_skips_llm(
        self, mock_fetch_single: patch, mock_fetch_batch: patch
    ) -> None:
        mock_fetch_batch.return_value = {"333": ("t3", "", "no_abstract")}
        candidates = pd.DataFrame(
            {
                "pmid": ["333"],
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "year": [2020],
            }
        )
        llm = _FakeLLM()
        out = run_unified_llm_filter(
            candidates,
            llm,
            verbose=0,
            abstract_cache_path=None,
            failed_retry_path=None,
            progress_every=0,
        )
        self.assertEqual(llm.prompts, [])
        self.assertEqual(out.loc[0, "unified_decision"], "uncertain")
        self.assertEqual(out.loc[0, "amr_primary_category"], "NONE")
        self.assertEqual(out.loc[0, "amr_evidence_phrases"], [])
        self.assertFalse(out.loc[0, "amr_target_pair_relation_supported"])
        self.assertFalse(out.loc[0, "amr_is_relevant"])
        self.assertEqual(mock_fetch_single.call_count, 0)


if __name__ == "__main__":
    unittest.main()
