# MedRAG – Production-Style Health AI RAG System

MedRAG is a retrieval-augmented generation (RAG) system for medical Q&A demonstration use-cases.  
It combines:
- **FAISS semantic retrieval** over a curated medical corpus
- **LLM generation** (Groq/Llama via API)
- **Evidence display + baseline comparison** for explainability
- **Lightweight Flask web app** for interactive demoing

> ⚠️ **Safety notice:** This project is for AI engineering demonstration and education. It is **not** a medical device and must not be used as clinical advice.

---

## Demo Screenshots

![MedRAG Architecture](figs/MedRAG.png)
![MedRAG Dashboard](figs/medbot_dashboard.png)
![MedRAG Demo](figs/medbot_demo.webp)

---

## Why this project matters

This repository demonstrates how to take an LLM proof-of-concept and shape it into a production-style system with:
- modular code (`src/`)
- explicit retrieval and grounding flow
- measurable quality metrics
- responsible-AI framing (factual grounding + explainability)

---

## Core capabilities

1. **Medical RAG pipeline**
   - Indexes medical text into FAISS.
   - Retrieves top-k evidence chunks per query.
   - Prompts the LLM with only retrieved context.

2. **Baseline vs RAG comparison**
   - Baseline answer (no retrieval context)
   - RAG answer (retrieval-grounded)
   - Side-by-side output to show measurable impact.

3. **Live evaluation metrics**
   - ROUGE-based overlap metrics
   - Latency tracking
   - Accuracy improvement signal from baseline vs RAG.

4. **Explainability-oriented UX**
   - Surfaces source snippets used for generation.
   - Encourages evidence-backed answers and transparency.

---

## High-level architecture

1. User sends question from web UI.
2. Flask API calls retrieval layer.
3. FAISS returns nearest medical evidence.
4. Generator produces:
   - grounded RAG answer
   - baseline non-grounded answer
5. Evaluator computes response metrics.
6. UI renders answers, sources, and metrics.

Primary modules:
- `app.py` – Flask API and runtime orchestration
- `src/retrieval/langchain_faiss_store.py` – vector index + retrieval
- `src/generation/llm_generators.py` – LLM inference wrappers
- `src/evaluation/metrics.py` – RAG quality metrics
- `src/data_loader.py` – dataset and corpus loading/fallbacks

---

## Tech stack

- **Backend:** Python, Flask
- **RAG orchestration:** LangChain
- **Vector store:** FAISS
- **Embeddings:** sentence-transformers
- **LLM API:** Groq-compatible chat model integration
- **Evaluation:** rouge-score + custom hallucination proxy metrics
- **Frontend:** HTML/CSS/JavaScript

---

## Quickstart

### 1) Clone and enter repo
```bash
git clone https://github.com/Flamechargerr/MedRAG.git
cd MedRAG
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment variables
Set at least one supported key:
- `GROQ_API_KEY` (preferred)
- `OPENCALL_LLM_KEY`
- `EMERGENT_LLM_KEY`

Example:
```bash
export GROQ_API_KEY="your_key_here"
```

### 4) Run the web app
```bash
python app.py
```

Open `http://127.0.0.1:5000`.

---

## Testing

Run lightweight unit tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run environment verification script:
```bash
python test_environment.py
```

---

## Responsible AI notes

- Retrieval-grounded prompting is used to reduce unsupported claims.
- Source snippets are returned to increase answer traceability.
- Baseline comparison highlights the value of grounding.
- This system should be treated as a decision-support prototype only.

---

## Resume / recruiter explanation guide

Use this short narrative:

> “I built a production-style medical RAG system with Flask, LangChain, FAISS, and LLM APIs.  
> The system compares baseline LLM responses against retrieval-grounded responses, exposes evidence snippets for explainability, and tracks live quality metrics like ROUGE and latency.  
> I designed it with responsible-AI principles for a sensitive healthcare context: grounding, transparency, and bias-awareness in evaluation.”

Suggested talking points:
- Why RAG in healthcare (factual grounding matters).
- Why FAISS (fast local semantic retrieval).
- How baseline vs RAG validated quality improvements.
- How explainability is implemented (source snippets + metrics).

---

## Limitations and next steps

- Add stronger clinical validation datasets and human expert review.
- Introduce automated CI for dependency, unit, and security checks.
- Add structured citation scoring and calibrated confidence outputs.
- Add role-based access controls and audit logging for real deployments.
