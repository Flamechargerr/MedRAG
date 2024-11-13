"""Integration tests for the MedRAG Flask application.

These tests use real dependency imports but stub heavy network operations.
They verify the actual routing, security, and service logic.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


class TestAppAPI(unittest.TestCase):
    """Test the Flask application API surface."""

    @classmethod
    def setUpClass(cls):
        # Ensure we can import from the repo root
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))

        # Set environment variables for test mode
        os.environ["APP_ENV"] = "local"
        os.environ["GROQ_API_KEY"] = "test-key"
        os.environ["RATE_LIMIT_PER_MINUTE"] = "200"
        os.environ["RUNTIME_STATE_PATH"] = str(tempfile.mktemp(suffix=".json"))

        # Import app after env setup
        from app import create_app
        cls.app = create_app()
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        # Clean up temp state file
        state_path = os.environ.get("RUNTIME_STATE_PATH")
        if state_path and Path(state_path).exists():
            Path(state_path).unlink()

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------
    def test_health_live(self):
        resp = self.client.get("/api/v1/health/live")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")

    def test_health_ready_before_init(self):
        resp = self.client.get("/api/v1/health/ready")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["status"], "initializing")

    def test_legacy_health_routes(self):
        resp = self.client.get("/health/live")
        self.assertEqual(resp.status_code, 200)
        resp = self.client.get("/health/ready")
        self.assertIn(resp.status_code, {200, 503})

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------
    def test_rejects_non_json_post(self):
        resp = self.client.post("/api/v1/chat", data="not json")
        self.assertEqual(resp.status_code, 415)
        self.assertEqual(resp.get_json()["status"], "error")

    def test_rejects_oversized_payload(self):
        # max_request_bytes is 64KB in default config
        large = {"query": "x" * (100 * 1024)}
        resp = self.client.post(
            "/api/v1/chat",
            data=json.dumps(large),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 413)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def test_init_rejects_invalid_corpus_size(self):
        resp = self.client.post(
            "/api/v1/init",
            json={"corpus_size": 0},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["status"], "error")

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def test_chat_rejects_before_init(self):
        resp = self.client.post(
            "/api/v1/chat",
            json={"query": "What is diabetes?"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["status"], "error")

    def test_chat_rejects_empty_query(self):
        # Initialize first (with minimal corpus)
        self.client.post("/api/v1/init", json={"corpus_size": 2})
        resp = self.client.post(
            "/api/v1/chat",
            json={"query": ""},
        )
        self.assertEqual(resp.status_code, 400)

    def test_chat_missing_query_field(self):
        resp = self.client.post("/api/v1/chat", json={})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Missing field", resp.get_json()["message"])

    # ------------------------------------------------------------------
    # 404 / 405
    # ------------------------------------------------------------------
    def test_404_returns_json(self):
        resp = self.client.get("/api/v1/nonexistent")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["status"], "error")

    def test_405_returns_json(self):
        resp = self.client.delete("/api/v1/health/live")
        self.assertEqual(resp.status_code, 405)
        self.assertEqual(resp.get_json()["status"], "error")


class TestSecurity(unittest.TestCase):
    """Test the RequestGuard security layer."""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))
        os.environ["APP_ENV"] = "local"

        from src.security import RequestGuard
        cls.guard_class = RequestGuard

    def _make_config(self, **kwargs):
        class C:
            pass
        c = C()
        c.app_auth_token = kwargs.get("token", "")
        c.max_request_bytes = kwargs.get("max_bytes", 64 * 1024)
        c.rate_limit_per_minute = kwargs.get("rate", 60)
        c.cors_origins = kwargs.get("cors", ["http://localhost:5000"])
        return c

    def test_request_size_rejected_when_too_large(self):
        guard = self.guard_class(self._make_config(max_bytes=10))
        # We can't fully test the guard without a Flask request context,
        # but we can test the internal logic via the method directly if we mock request
        import flask
        with flask.Flask(__name__).test_request_context(
            method="POST", content_type="application/json"
        ):
            flask.request.content_length = 100
            result = guard.enforce_request_size()
            self.assertIsNotNone(result)
            payload, status = result
            self.assertEqual(status, 413)

    def test_rate_limit_exceeded(self):
        guard = self.guard_class(self._make_config(rate=2))
        import flask
        with flask.Flask(__name__).test_request_context(
            method="POST", path="/api/v1/chat", content_type="application/json"
        ):
            flask.request.content_length = 0
            # First 2 requests pass
            self.assertIsNone(guard.enforce_rate_limit())
            self.assertIsNone(guard.enforce_rate_limit())
            # Third is blocked
            result = guard.enforce_rate_limit()
            self.assertIsNotNone(result)
            payload, status = result
            self.assertEqual(status, 429)


class TestMetrics(unittest.TestCase):
    """Test the honest evaluation metrics."""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))
        from src.evaluation.metrics import CitationGroundingEvaluator
        cls.evaluator = CitationGroundingEvaluator()

    def test_grounding_high_with_overlap(self):
        docs = [type("D", (), {"text": "insulin treats diabetes effectively by regulating blood glucose"})]
        answer = "Insulin is used to treat diabetes by regulating blood glucose levels."
        grounding = self.evaluator.compute_grounding(answer, docs)
        self.assertGreater(grounding, 0.5)

    def test_grounding_zero_with_no_overlap(self):
        docs = [type("D", (), {"text": "aspirin is used for pain relief"})]
        answer = "The sky is blue and water is wet."
        grounding = self.evaluator.compute_grounding(answer, docs)
        self.assertEqual(grounding, 0.0)

    def test_confidence_high_for_definitive_answer(self):
        docs = []
        answer = "According to clinical guidelines, metformin is first-line for type 2 diabetes."
        conf = self.evaluator.estimate_answer_confidence(answer, docs)
        self.assertGreater(conf, 0.7)

    def test_confidence_low_for_hedging_answer(self):
        docs = []
        answer = "Maybe possibly perhaps the treatment could be effective, but it is unclear."
        conf = self.evaluator.estimate_answer_confidence(answer, docs)
        self.assertLess(conf, 0.6)

    def test_rouge_scores_sanity(self):
        scores = self.evaluator.compute_rouge_scores("the cat sat on the mat", "the cat sat on the mat")
        self.assertEqual(scores["rouge1"], 1.0)
        self.assertEqual(scores["rougeL"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
