import importlib
import sys
import types
import unittest


def _install_rouge_stub():
    rouge_module = types.ModuleType("rouge_score")

    class _Score:
        def __init__(self, value):
            self.fmeasure = value

    class RougeScorer:
        def __init__(self, metrics, use_stemmer=True):
            self.metrics = metrics

        def score(self, reference, generated):
            return {"rouge1": _Score(0.6), "rouge2": _Score(0.5), "rougeL": _Score(0.55)}

    rouge_module.rouge_scorer = types.SimpleNamespace(RougeScorer=RougeScorer)
    sys.modules["rouge_score"] = rouge_module


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_rouge_stub()
        sys.modules.pop("src.evaluation.metrics", None)
        cls.metrics_module = importlib.import_module("src.evaluation.metrics")
        cls.evaluator = cls.metrics_module.RAGEvaluator()

    def test_compute_retrieval_metrics_with_overlap(self):
        retrieved = [{"id": "a"}, {"id": "b"}]
        ground_truth = [{"id": "b"}, {"id": "c"}]
        scores = self.evaluator.compute_retrieval_metrics(retrieved, ground_truth)
        self.assertAlmostEqual(scores["precision"], 0.5)
        self.assertAlmostEqual(scores["recall"], 0.5)
        self.assertAlmostEqual(scores["f1"], 0.5)

    def test_detect_hallucination_returns_low_when_overlap_high(self):
        docs = [{"text": "insulin treats diabetes effectively"}]
        hallucination_rate = self.evaluator.detect_hallucination(
            "insulin treats diabetes", docs
        )
        self.assertLess(hallucination_rate, 0.34)

    def test_measure_response_time_returns_result_and_duration(self):
        result, duration = self.evaluator.measure_response_time(lambda x: x + 1, 2)
        self.assertEqual(result, 3)
        self.assertGreaterEqual(duration, 0.0)

    def test_compute_rouge_scores_uses_stubbed_values(self):
        scores = self.evaluator.compute_rouge_scores("generated", "reference")
        self.assertEqual(scores["rouge1"], 0.6)
        self.assertEqual(scores["rouge2"], 0.5)
        self.assertEqual(scores["rougeL"], 0.55)


if __name__ == "__main__":
    unittest.main()
