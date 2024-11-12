# MedRAG Evaluation Methodology

## Purpose

This document describes how MedRAG outputs are evaluated for research and benchmarking purposes. These metrics are for **comparative analysis** of retrieval-augmented generation systems, not for clinical validation.

## Metrics

### 1. Citation Grounding

**Definition:** The fraction of sentences in the generated answer that have meaningful token overlap with the retrieved evidence documents.

**How it works:**
- Split answer into sentences.
- For each sentence, compute token overlap with the union of all retrieved document tokens.
- A sentence is "grounded" if overlap ≥ 30%.
- Grounding = (grounded sentences) / (total sentences).

**Limitations:**
- Token overlap is a proxy, not semantic understanding.
- Paraphrased or implied facts may not register as overlap.
- Does not guarantee clinical correctness.

### 2. Hallucination Risk

**Definition:** 1.0 − citation_grounding. A higher value means more of the answer is not directly supported by retrieved evidence.

**Important:** This is a **retrieval-grounded** metric, not a clinical fact-check. An answer may be factually correct but have low grounding if the retrieval corpus is incomplete. Conversely, an answer may be grounded but wrong if the retrieved evidence itself is outdated or incorrect.

### 3. Answer Confidence Heuristic

**Definition:** A structural/lexical estimate of the LLM's apparent confidence based on:
- Hedging words ("maybe", "possibly", "unclear") → penalty
- Evidence phrases ("according to", "studies show") → bonus
- Citation markers → bonus
- Answer length (too short or too long) → penalty

**Limitations:** Measures linguistic style, not actual model uncertainty. A confident-sounding wrong answer scores high.

### 4. ROUGE Scores

**Definition:** Standard ROUGE-1, ROUGE-2, ROUGE-L F1 scores between generated answer and a user-provided reference answer.

**How we report them:**
- `rag_rouge_l`: ROUGE-L of the RAG-generated answer
- `baseline_rouge_l`: ROUGE-L of the no-retrieval baseline
- `rouge_delta`: Raw difference (RAG − Baseline)

**We do NOT report:** A misleading "accuracy improvement" percentage. ROUGE measures lexical overlap, not clinical accuracy. A 0.15 difference in ROUGE-L does not mean 15% better patient outcomes.

### 5. Retrieval Confidence

**Definition:** A normalized score indicating the quality of the retrieved documents.
- For MedRAG (RRF): based on top RRF score
- For FAISS fallback: based on average normalized distance

## Benchmarking Workflow

1. Load evaluation questions (MedQA-USMLE or MedMCQA).
2. For each question, run both RAG and baseline pipelines.
3. Compute all metrics above.
4. Report **aggregate statistics** with confidence intervals where possible.
5. Never claim clinical superiority based solely on ROUGE scores.

## What These Metrics Cannot Tell You

- **Clinical correctness:** An answer can be grounded and wrong.
- **Safety:** An answer can be confident and harmful.
- **Generalizability:** Results on MedQA may not transfer to real-world clinical scenarios.
- **Causal inference:** RAG may improve ROUGE but not patient outcomes.

## Responsible Use

- Use these metrics for **system development** and **research comparison** only.
- Do not use them to make claims about clinical efficacy.
- Always involve medical professionals in validation.
- Cite the original MedRAG paper when publishing results.
