#!/usr/bin/env python3
"""MedRAG Evaluation CLI — Benchmark the RAG pipeline against a question set.

Usage:
    python scripts/main.py --demo        # Quick test with 2 questions
    python scripts/main.py               # Full evaluation with 100 questions

Outputs aggregate metrics to stdout and detailed per-question results to
results/evaluation_results.json.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, Config
from src.core.engine import MedRAGEngine
from src.data_loader import load_medqa_data, load_medical_corpus
from src.evaluation.metrics import CitationGroundingEvaluator
from src.generation.llm_generators import GroqGenerator
from src.retrieval.langchain_faiss_store import LangchainFAISSStore

logger = logging.getLogger("medrag-eval")


def run_evaluation(
    system_name: str,
    questions,
    retrieve_func,
    generate_func,
    baseline_generate_func,
    evaluator: CitationGroundingEvaluator,
):
    logger.info("--- Evaluating: %s ---", system_name)
    results = []

    for item in questions:
        if isinstance(item, dict):
            question = item.get("question", item.get("sent1", ""))
            reference = item.get("answer", item.get("ending0", "Unknown"))
        else:
            question = str(item)
            reference = "Unknown"

        if not question:
            continue

        # RAG pipeline
        retrieved_docs, r_time = evaluator.measure_response_time(retrieve_func, question)
        rag_answer, g_time = evaluator.measure_response_time(generate_func, question, retrieved_docs)

        # Baseline (no retrieval)
        baseline_answer, b_time = evaluator.measure_response_time(baseline_generate_func, question)

        # Metrics
        rag_scores = evaluator.compute_rouge_scores(rag_answer, reference)
        base_scores = evaluator.compute_rouge_scores(baseline_answer, reference)
        grounding = evaluator.compute_grounding(rag_answer, retrieved_docs)
        confidence = evaluator.estimate_answer_confidence(rag_answer, retrieved_docs)

        total_time = r_time + g_time

        results.append({
            "system": system_name,
            "question": question[:120],
            "reference": reference,
            "rag_answer": rag_answer,
            "baseline_answer": baseline_answer,
            "rouge1": round(rag_scores["rouge1"], 4),
            "rougeL": round(rag_scores["rougeL"], 4),
            "baseline_rougeL": round(base_scores["rougeL"], 4),
            "rouge_delta": round(rag_scores["rougeL"] - base_scores["rougeL"], 4),
            "citation_grounding": grounding,
            "hallucination_risk": round(1.0 - grounding, 3),
            "answer_confidence": confidence,
            "latency_seconds": round(total_time, 3),
            "n_sources": len(retrieved_docs),
        })

    # Aggregate
    n = len(results)
    if n == 0:
        logger.warning("No results collected.")
        return results

    avg = lambda key: sum(r[key] for r in results) / n
    logger.info("Results for %s:", system_name)
    logger.info("  Avg ROUGE-L:        %.4f", avg("rougeL"))
    logger.info("  Avg Baseline R-L:   %.4f", avg("baseline_rougeL"))
    logger.info("  Avg Rouge Delta:    %+.4f", avg("rouge_delta"))
    logger.info("  Avg Grounding:      %.3f", avg("citation_grounding"))
    logger.info("  Avg Hallucination:  %.3f", avg("hallucination_risk"))
    logger.info("  Avg Confidence:     %.3f", avg("answer_confidence"))
    logger.info("  Avg Latency:        %.3fs", avg("latency_seconds"))

    return results


def main():
    parser = argparse.ArgumentParser(description="MedRAG Evaluation CLI")
    parser.add_argument("--demo", action="store_true", help="Quick demo with 2 questions")
    parser.add_argument("--output", default="results/evaluation_results.json", help="Output JSON path")
    parser.add_argument("--corpus", default="Textbooks", help="Corpus name (Textbooks, PubMed, StatPearls, etc.)")
    parser.add_argument("--retriever", default="MedCPT", help="Retriever name (MedCPT, BM25, Contriever, etc.)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    Config.load_env()
    config = get_config()
    logger.info("Running on device: %s", config.device)

    eval_size = 2 if args.demo else 100
    corpus_size = 50 if args.demo else 2000

    # 1. Load data
    eval_questions, full_dataset = load_medqa_data(num_eval_questions=eval_size)

    # 2. Initialize engine
    engine = MedRAGEngine(
        db_dir="./corpus",
        corpus_name=args.corpus,
        retriever_name=args.retriever,
        faiss_fallback_dir="./faiss_db",
        device=config.device,
        retrieval_top_k=5,
    )
    init_result = engine.initialize_blocking()
    logger.info("Engine init: %s", init_result.get("message"))

    # 3. LLM
    llm_gen = GroqGenerator(config.api_key, max_retries=config.llm_max_retries)

    # 4. Evaluator
    evaluator = CitationGroundingEvaluator()

    # 5. Run evaluation
    def rag_retriever(q):
        return engine.retrieve(q)

    def rag_generator(q, docs):
        return llm_gen.generate(q, docs)

    def baseline_generator(q):
        return llm_gen.generate_no_context(q)

    results = run_evaluation(
        system_name=f"MedRAG ({args.retriever} + {args.corpus})",
        questions=eval_questions,
        retrieve_func=rag_retriever,
        generate_func=rag_generator,
        baseline_generate_func=baseline_generator,
        evaluator=evaluator,
    )

    # 6. Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "corpus": args.corpus,
                    "retriever": args.retriever,
                    "n_questions": len(results),
                    "device": config.device,
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
