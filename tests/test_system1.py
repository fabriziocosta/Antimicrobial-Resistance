import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from amr.screening import (
    OpenRouterJevClient,
    build_yearly_screened_pairs,
    run_system1_filter,
)


class _FakeDecisionClient:
    def __init__(self, response=None, error=None):
        self.response = response or {
            "model": "typesafe/jev-1.13-20260917",
            "provider": "TypeSafe",
            "id": "req-1",
            "answers": {"amr_pair_supported": {"type": "noul", "noul": 0.82}},
        }
        self.error = error
        self.calls = 0

    def decide(self, state, *, question):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


class _FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class TestSystem1Predictions(unittest.TestCase):
    def setUp(self):
        self.candidates = pd.DataFrame(
            [{"pmid": "101", "pathogen": "E. coli", "antibiotic": "Ciprofloxacin", "year": 2022}]
        )
        self.abstracts = {"101": ("Resistance in E. coli", "Ciprofloxacin resistance was observed.", "ok")}

    @patch("amr.screening.system1.fetch_pubmed_abstracts_batch")
    def test_records_probability_and_reuses_completed_run(self, fetch):
        fetch.return_value = self.abstracts
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "system1.parquet"
            client = _FakeDecisionClient()
            first = run_system1_filter(
                self.candidates, client, run_id="run-a", output_path=output, abstract_cache_path=None
            )
            self.assertEqual(first.loc[0, "system1_p_yes"], 0.82)
            self.assertEqual(first.loc[0, "system1_status"], "ok")
            self.assertEqual(first.loc[0, "system1_served_model"], "typesafe/jev-1.13-20260917")

            second_client = _FakeDecisionClient()
            second = run_system1_filter(
                self.candidates, second_client, run_id="run-a", output_path=output, abstract_cache_path=None
            )
            self.assertEqual(second_client.calls, 0)
            self.assertEqual(len(second), 1)
            self.assertEqual(len(pd.read_parquet(output)), 1)
            with self.assertRaisesRegex(ValueError, "new run_id"):
                run_system1_filter(
                    self.candidates,
                    _FakeDecisionClient(),
                    model="typesafe/jev-next",
                    run_id="run-a",
                    output_path=output,
                    abstract_cache_path=None,
                )

    @patch("amr.screening.system1.fetch_pubmed_abstracts_batch")
    def test_bad_probability_and_api_failure_are_recorded_without_replacing_prior_run(self, fetch):
        fetch.return_value = self.abstracts
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "system1.parquet"
            run_system1_filter(
                self.candidates,
                _FakeDecisionClient(),
                run_id="run-good",
                output_path=output,
                abstract_cache_path=None,
            )
            malformed = _FakeDecisionClient(
                response={"answers": {"amr_pair_supported": {"type": "noul", "noul": 1.2}}}
            )
            failed = run_system1_filter(
                self.candidates,
                malformed,
                run_id="run-bad",
                output_path=output,
                abstract_cache_path=None,
            )
            self.assertEqual(failed.loc[0, "system1_status"], "error")
            self.assertIn("between 0 and 1", failed.loc[0, "system1_error"])
            stored = pd.read_parquet(output)
            self.assertEqual(set(stored["system1_run_id"]), {"run-good", "run-bad"})
            self.assertEqual(stored.loc[stored.system1_run_id == "run-good", "system1_p_yes"].iloc[0], 0.82)

            missing_answer = _FakeDecisionClient(response={"answers": {}})
            missing = run_system1_filter(
                self.candidates,
                missing_answer,
                run_id="run-missing-answer",
                output_path=output,
                abstract_cache_path=None,
            )
            self.assertEqual(missing.loc[0, "system1_status"], "error")
            self.assertIn("missing the AMR noul answer", missing.loc[0, "system1_error"])

            api_failure = _FakeDecisionClient(error=RuntimeError("upstream unavailable"))
            retry = run_system1_filter(
                self.candidates,
                api_failure,
                run_id="run-api-error",
                output_path=output,
                abstract_cache_path=None,
            )
            self.assertEqual(retry.loc[0, "system1_status"], "error")
            self.assertIn("upstream unavailable", retry.loc[0, "system1_error"])
            self.assertEqual(len(pd.read_parquet(output)), 4)

    def test_openrouter_client_posts_typed_noul_question(self):
        session = Mock()
        session.post.return_value = _FakeResponse({"answers": {}})
        client = OpenRouterJevClient(api_key="test-key", session=session, retries=1)
        client.decide("paper state", question="Is the AMR pair supported?")
        _, kwargs = session.post.call_args
        self.assertEqual(kwargs["json"]["model"], "typesafe/jev-1.13")
        self.assertEqual(kwargs["json"]["questions"]["amr_pair_supported"]["type"], "noul")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")

    @patch("amr.screening.system1.fetch_pubmed_abstracts_batch")
    def test_system1_runs_requests_concurrently_when_configured(self, fetch):
        candidates = pd.DataFrame(
            [
                {"pmid": str(pmid), "pathogen": "E. coli", "antibiotic": "Ciprofloxacin", "year": 2022}
                for pmid in range(4)
            ]
        )
        fetch.return_value = {
            str(pmid): ("title", "abstract", "ok") for pmid in range(4)
        }

        class ConcurrentClient(_FakeDecisionClient):
            def __init__(self):
                super().__init__()
                self.guard = threading.Lock()
                self.active = 0
                self.max_active = 0

            def decide(self, state, *, question):
                with self.guard:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.03)
                with self.guard:
                    self.active -= 1
                return self.response

        client = ConcurrentClient()
        results = run_system1_filter(
            candidates,
            client,
            run_id="parallel-run",
            abstract_cache_path=None,
            max_workers=4,
            progress_every=0,
        )
        self.assertEqual(len(results), 4)
        self.assertTrue(results["system1_status"].eq("ok").all())
        self.assertGreater(client.max_active, 1)

    @patch("amr.screening.system1.fetch_pubmed_abstracts_batch")
    def test_credit_forecast_waits_for_reported_spend(self, fetch):
        candidates = pd.DataFrame(
            [
                {"pmid": str(pmid), "pathogen": "E. coli", "antibiotic": "Ciprofloxacin", "year": 2022}
                for pmid in range(2)
            ]
        )
        fetch.return_value = {str(pmid): ("title", "abstract", "ok") for pmid in range(2)}

        class CreditClient(_FakeDecisionClient):
            def __init__(self):
                super().__init__()
                self.balances = iter([10.0, 10.0, 9.9, 9.9])

            def credit_status(self):
                balance = next(self.balances)
                return f"account credits remaining ${balance:.4f}", balance

        output = StringIO()
        with redirect_stdout(output):
            run_system1_filter(
                candidates,
                CreditClient(),
                run_id="credit-run",
                abstract_cache_path=None,
                progress_every=1,
            )
        lines = output.getvalue().splitlines()
        self.assertIn("end balance estimate waiting for OpenRouter to report spend", lines[1])
        self.assertIn("estimated balance after 0 unresolved candidates $9.9000", lines[2])

    def test_system1_threshold_and_run_selection(self):
        yearly = pd.DataFrame(
            [{"pathogen": "E. coli", "antibiotic": "Ciprofloxacin", "year": 2022, "pmids": ["101", "102"]}]
        )
        predictions = pd.DataFrame(
            [
                {"pmid": "101", "pathogen": "E. coli", "antibiotic": "Ciprofloxacin", "year": 2022,
                 "system1_run_id": "run-a", "system1_p_yes": 0.82, "system1_status": "ok"},
                {"pmid": "102", "pathogen": "E. coli", "antibiotic": "Ciprofloxacin", "year": 2022,
                 "system1_run_id": "run-a", "system1_p_yes": 0.49, "system1_status": "ok"},
                {"pmid": "102", "pathogen": "E. coli", "antibiotic": "Ciprofloxacin", "year": 2022,
                 "system1_run_id": "run-b", "system1_p_yes": 1.0, "system1_status": "ok"},
            ]
        )
        selected = build_yearly_screened_pairs(
            yearly, predictions, prediction_source="system1", system1_run_id="run-a", probability_threshold=0.5
        )
        self.assertEqual(selected.loc[0, "pmids"], ["101"])
        self.assertEqual(selected.loc[0, "count"], 1)

        lower_cutoff = build_yearly_screened_pairs(
            yearly, predictions, prediction_source="system1", system1_run_id="run-a", probability_threshold=0.4
        )
        self.assertEqual(set(lower_cutoff.loc[0, "pmids"]), {"101", "102"})

        with self.assertRaisesRegex(ValueError, "required when multiple"):
            build_yearly_screened_pairs(
                yearly, predictions, prediction_source="system1", probability_threshold=0.5
            )


if __name__ == "__main__":
    unittest.main()
