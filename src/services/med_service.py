"""MedRAGService — Clean orchestration layer for the web API.

Uses the real MedRAGEngine (MedCPT + multi-corpus RRF) when available, with a
lightweight FAISS fallback. Provides honest, clinically-relevant metrics.
"""

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from src.core.engine import MedRAGEngine, RetrievedDocument
from src.evaluation.metrics import CitationGroundingEvaluator
from src.generation.llm_generators import GroqGenerator

logger = logging.getLogger(__name__)


class MedRAGService:
    """Production orchestrator: retrieval → generation → honest evaluation."""

    def __init__(self, config):
        self.config = config
        self.engine = MedRAGEngine(
            db_dir="./corpus",
            corpus_name=getattr(config, "corpus_name", "Textbooks"),
            retriever_name=getattr(config, "retriever_name", "MedCPT"),
            faiss_fallback_dir=getattr(config, "faiss_db_dir", "./faiss_db"),
            device=getattr(config, "device", "cpu"),
            state_path=getattr(config, "runtime_state_path", "./runtime_state.json"),
            retrieval_top_k=getattr(config, "retrieval_top_k", 5),
        )
        self.llm_gen: Optional[GroqGenerator] = None
        self.evaluator = CitationGroundingEvaluator()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self, corpus_size: int, force_reindex: bool = False) -> Dict[str, Any]:
        """Initialize the system.  corpus_size is advisory for fallback mode."""
        if self.engine.health_ready() and not force_reindex:
            return {"status": "success", "message": "System already initialized."}
        return self.engine.initialize_blocking()

    def health_ready(self) -> bool:
        return self.engine.health_ready()

    def get_status(self) -> Dict[str, Any]:
        return self.engine.get_init_status()

    # ------------------------------------------------------------------
    # Chat / RAG pipeline
    # ------------------------------------------------------------------
    def chat(self, question: str, reference: str = "") -> Dict[str, Any]:
        if not self.engine.health_ready():
            raise RuntimeError("System is not initialized")
        if not question or not question.strip():
            raise ValueError("No query provided")
        if len(question) > getattr(self.config, "max_query_chars", 2000):
            raise ValueError(f"Query exceeds max length ({self.config.max_query_chars} characters)")

        start_time = time.time()

        # 1. Retrieve
        docs = self.engine.retrieve(question)
        retrieval_confidence = self.engine.compute_retrieval_confidence(docs)

        # 2. Build grounded prompt
        context = self._build_context(docs)

        # 3. Initialize LLM on first use (lazy)
        if self.llm_gen is None:
            self.llm_gen = GroqGenerator(
                api_key=self.config.api_key,
                max_retries=getattr(self.config, "llm_max_retries", 2),
            )

        # 4. Generate RAG answer
        rag_answer = self.llm_gen.generate(question, docs)

        # 5. Generate baseline (no-context) for comparison
        baseline_answer = self.llm_gen.generate_no_context(question)

        latency = time.time() - start_time

        # 6. Evaluate — honest, clinically-relevant metrics
        metrics = self._evaluate(
            rag_answer=rag_answer,
            baseline_answer=baseline_answer,
            docs=docs,
            context=context,
            reference=reference,
            latency=latency,
            retrieval_confidence=retrieval_confidence,
        )

        # 7. Format sources
        sources = self._format_sources(docs)

        return {
            "status": "success",
            "answer": rag_answer,
            "baseline_answer": baseline_answer,
            "sources": sources,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_context(self, docs: List[RetrievedDocument]) -> str:
        """Build a citation-rich context block for the LLM."""
        parts = []
        for d in docs:
            parts.append(f"[{d.rank}] {d.title} (Source: {d.source})")
            parts.append(d.text)
            parts.append("")
        return "\n".join(parts)

    def _format_sources(self, docs: List[RetrievedDocument]) -> List[Dict[str, Any]]:
        """Format sources for the API response."""
        return [
            {
                "id": d.id,
                "title": d.title,
                "text": d.text[:500] + ("..." if len(d.text) > 500 else ""),
                "source": d.source,
                "score": round(d.score, 4),
                "rank": d.rank,
            }
            for d in docs
        ]

    def _evaluate(
        self,
        rag_answer: str,
        baseline_answer: str,
        docs: List[RetrievedDocument],
        context: str,
        reference: str,
        latency: float,
        retrieval_confidence: float,
    ) -> Dict[str, Any]:
        """Compute honest evaluation metrics."""
        metrics: Dict[str, Any] = {}

        # Timing
        metrics["latency_ms"] = int(latency * 1000)
        metrics["latency"] = f"{latency:.2f}s"

        # Retrieval quality
        metrics["retrieval_confidence"] = retrieval_confidence
        metrics["n_sources"] = len(docs)
        metrics["source_coverage"] = list(set(d.source for d in docs)) if docs else []

        # Citation grounding — how much of the answer is supported by retrieved docs?
        grounding = self.evaluator.compute_grounding(rag_answer, docs)
        metrics["citation_grounding"] = grounding  # 0-1, fraction of claims supported
        metrics["hallucination_risk"] = round(1.0 - grounding, 3)

        # ROUGE against reference (if user provided one)
        if reference:
            rag_scores = self.evaluator.compute_rouge_scores(rag_answer, reference)
            base_scores = self.evaluator.compute_rouge_scores(baseline_answer, reference)
            metrics["rag_rouge_l"] = round(rag_scores["rougeL"], 4)
            metrics["baseline_rouge_l"] = round(base_scores["rougeL"], 4)
            # Report raw scores, NOT a misleading "accuracy improvement" percentage
            metrics["rouge_delta"] = round(rag_scores["rougeL"] - base_scores["rougeL"], 4)

        # Answer confidence heuristic
        metrics["answer_confidence"] = self.evaluator.estimate_answer_confidence(
            rag_answer, docs
        )

        return metrics
