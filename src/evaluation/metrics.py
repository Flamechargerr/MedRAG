"""Honest evaluation metrics for MedRAG.

Replaces the gimmicky ROUGE-derived "accuracy improvement" with clinically-meaningful
metrics: citation grounding, hallucination risk, and answer confidence heuristics.
"""

import re
import time
from typing import Dict, List, Any

from rouge_score import rouge_scorer


class CitationGroundingEvaluator:
    """Evaluate RAG outputs with clinically-relevant, honest metrics."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    # ------------------------------------------------------------------
    # ROUGE (for reference comparison, not misleading "accuracy")
    # ------------------------------------------------------------------
    def compute_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
        scores = self.rouge_scorer.score(str(reference), str(generated))
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    # ------------------------------------------------------------------
    # Citation grounding (the core honest metric)
    # ------------------------------------------------------------------
    def compute_grounding(self, answer: str, docs: List[Any]) -> float:
        """Estimate what fraction of the answer is supported by retrieved evidence.

        Heuristic: tokenize answer into sentences, check each sentence against
        the union of retrieved doc tokens.  Return fraction of sentences with
        meaningful overlap.
        """
        if not answer or not docs:
            return 0.0

        sentences = self._split_sentences(answer)
        if not sentences:
            return 0.0

        # Build vocabulary from all retrieved docs
        doc_tokens = set()
        for doc in docs:
            text = doc.text if hasattr(doc, "text") else doc.get("text", "")
            doc_tokens.update(self._tokenize(text))

        supported = 0
        for sentence in sentences:
            sent_tokens = self._tokenize(sentence)
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & doc_tokens) / len(sent_tokens)
            if overlap >= 0.3:  # 30% token overlap threshold
                supported += 1

        return round(supported / len(sentences), 3) if sentences else 0.0

    # ------------------------------------------------------------------
    # Answer confidence heuristics
    # ------------------------------------------------------------------
    def estimate_answer_confidence(self, answer: str, docs: List[Any]) -> float:
        """Estimate LLM answer confidence using structural and lexical heuristics.

        - High confidence: definitive language, citations present, length appropriate
        - Low confidence: hedging words, no citations, very short / very long
        """
        if not answer:
            return 0.0

        hedging_words = [
            "maybe", "possibly", "perhaps", "might", "could", "unclear",
            "unknown", "insufficient", "not sure", "likely", "probably",
            "suggest", "appears to", "seems to",
        ]
        confidence_markers = [
            "based on", "according to", "evidence suggests", "studies show",
            "clinical guidelines", "systematic review", "meta-analysis",
        ]

        text_lower = answer.lower()

        # Hedging penalty
        hedge_count = sum(1 for w in hedging_words if w in text_lower)
        hedge_penalty = min(0.4, hedge_count * 0.08)

        # Confidence bonus
        conf_bonus = min(0.2, sum(1 for w in confidence_markers if w in text_lower) * 0.05)

        # Citation bonus (bracket numbers or "Source:" markers)
        citation_bonus = 0.1 if re.search(r"\[\d+\]|Source:|source:|citation", text_lower) else 0.0

        # Length penalty (too short or too long)
        words = len(answer.split())
        if words < 20:
            length_penalty = 0.15
        elif words > 400:
            length_penalty = 0.1
        else:
            length_penalty = 0.0

        base = 0.75
        score = base - hedge_penalty + conf_bonus + citation_bonus - length_penalty
        return round(max(0.0, min(1.0, score)), 3)

    # ------------------------------------------------------------------
    # Retrieval metrics (for evaluation mode, not chat)
    # ------------------------------------------------------------------
    def compute_retrieval_metrics(
        self, retrieved_docs: List[Any], ground_truth_docs: List[Any]
    ) -> Dict[str, float]:
        """Precision / Recall / F1 at the document level."""
        retrieved_ids = set(
            doc.id if hasattr(doc, "id") else doc.get("id", "")
            for doc in retrieved_docs
        )
        ground_truth_ids = set(
            doc.id if hasattr(doc, "id") else doc.get("id", "")
            for doc in ground_truth_docs
        )
        if not ground_truth_ids:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        tp = len(retrieved_ids & ground_truth_ids)
        precision = tp / len(retrieved_ids) if retrieved_ids else 0.0
        recall = tp / len(ground_truth_ids) if ground_truth_ids else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def measure_response_time(self, func, *args, **kwargs):
        """Measure wall-clock time of a function call."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    # ------------------------------------------------------------------
    # Internal tokenization
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> set:
        """Simple tokenization: lowercase words, remove short tokens."""
        return {
            w.strip(".,;:!?()[]{}'\"")
            for w in text.lower().split()
            if len(w.strip(".,;:!?()[]{}'\"")) > 2
        }

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]
