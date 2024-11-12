# MedRAG

[![CI](https://github.com/Flamechargerr/MedRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Flamechargerr/MedRAG/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-oriented implementation of **MedRAG** — Retrieval-Augmented Generation for medical literature. Based on the research by [Xiong et al., 2024](https://github.com/RUCAIBox/MedRAG), this system retrieves relevant passages from published medical corpora and grounds LLM responses in that evidence, providing transparent citation metrics and hallucination risk assessment.

> **⚠️ Research Use Only.** This system retrieves and synthesizes published medical literature. It does not provide clinical diagnosis, treatment advice, or patient care recommendations. Always verify outputs with primary sources and qualified medical professionals.

---

## Screenshots

### Query Interface & RAG Response
The clean, professional web interface shows the RAG response alongside the baseline (no-retrieval) response, with real-time metrics including citation grounding, hallucination risk, and retrieval confidence.

![MedRAG UI - Query and Response](https://raw.githubusercontent.com/Flamechargerr/MedRAG/main/docs/assets/ui_preview_top.png)

### Retrieved Sources with Provenance
Each source is displayed with its title, corpus origin (PubMed, Textbooks, StatPearls), and retrieval score, enabling transparent verification of the evidence.

![MedRAG UI - Retrieved Sources](https://raw.githubusercontent.com/Flamechargerr/MedRAG/main/docs/assets/ui_preview_sources.png)

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Running the System](#running-the-system)
6. [API Reference](#api-reference)
7. [Metrics & Evaluation](#metrics--evaluation)
8. [Deployment](#deployment)
9. [Testing](#testing)
10. [Environment Variables](#environment-variables)
11. [Troubleshooting](#troubleshooting)
12. [License & Citation](#license--citation)

---

## What This Is

MedRAG is a **medical literature retrieval and synthesis framework** designed for research and benchmarking. It performs three core operations:

1. **Retrieval** — Given a medical question, the system retrieves the most relevant passages from medical corpora using domain-specific bi-encoders (MedCPT, SPECTER, Contriever) and Reciprocal Rank Fusion (RRF).
2. **Synthesis** — A language model generates an answer constrained to the retrieved evidence via a structured prompt.
3. **Evaluation** — The system reports honest, clinically-relevant metrics: citation grounding fraction, hallucination risk, retrieval confidence, and source provenance.

### What This Is NOT

- ❌ A medical AI assistant or chatbot
- ❌ A diagnostic tool or clinical decision support system
- ❌ A certified medical device
- ❌ A substitute for qualified medical professionals

### Supported Corpora

| Corpus | Description | Size |
|--------|-------------|------|
| **PubMed** | Abstracts from MEDLINE | ~36M documents |
| **Textbooks** | Medical textbooks (HuggingFace MedRAG) | ~3K documents |
| **StatPearls** | NCBI StatPearls review articles | ~7K documents |
| **Wikipedia** | Medical Wikipedia articles | ~60K documents |

### Supported Retrievers

| Retriever | Description | Use Case |
|-----------|-------------|----------|
| **MedCPT** | Bi-encoder trained on PubMed query-article pairs | **Default** — best for medical queries |
| **BM25** | Sparse lexical retriever | Fast, no GPU needed |
| **SPECTER** | Scientific paper embeddings | Good for academic literature |
| **Contriever** | Unsupervised contrastive embeddings | General-purpose dense retrieval |
| **RRF-2/4** | Ensemble of multiple retrievers | Maximum recall |

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────┐
│   Query     │────▶│   MedRAG     │────▶│  Retrieved Documents        │
│   (user)    │     │   Engine     │     │  (PubMed / Textbooks /     │
└─────────────┘     └──────────────┘     │   StatPearls / Wikipedia)   │
                            │            └─────────────────────────────┘
                            ▼                            │
                     ┌──────────────┐                     │
                     │  LLM (Groq)  │◄────────────────────┘
                     │  + Prompt    │   (context + question)
                     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Response    │
                     │  + Metrics   │
                     └──────────────┘
```

### Retrieval Backends

| Mode | Description | When Used |
|------|-------------|-----------|
| **Full MedRAG** | MedCPT + multi-corpus + RRF fusion | When corpora are downloaded and indexed |
| **Fallback FAISS** | PubMedBERT embeddings + lightweight FAISS | When full corpora are unavailable |

The engine attempts **Full MedRAG** first and automatically falls back to the lightweight mode with a clear status report.

### Retrieval Pipeline (Full Mode)

```
Query ──▶ MedCPT Query Encoder ──▶ FAISS Index ──▶ Top-K Docs
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
               [PubMed]          [Textbooks]         [StatPearls]
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        ▼
                              Reciprocal Rank Fusion
                                        │
                                        ▼
                              Deduplicated, Ranked Results
```

---

## Project Structure

```
MedRAG/
├── src/
│   ├── core/
│   │   └── engine.py              # Production engine: MedRAG + fallback
│   ├── evaluation/
│   │   └── metrics.py             # Citation grounding, hallucination risk, ROUGE
│   ├── generation/
│   │   └── llm_generators.py      # Groq LLM with retry + fallback
│   ├── retrieval/
│   │   ├── langchain_faiss_store.py  # Fallback FAISS vector store
│   │   └── vector_store.py          # ChromaDB (legacy)
│   ├── services/
│   │   └── med_service.py         # Flask service orchestrator
│   ├── data/
│   │   ├── pubmed.py              # PubMed corpus loader
│   │   ├── statpearls.py          # StatPearls corpus loader
│   │   ├── textbooks.py           # Medical textbooks loader
│   │   └── wikipedia.py           # Wikipedia medical loader
│   ├── data_loader.py             # Unified corpus loading
│   ├── config.py                  # Environment configuration
│   ├── security.py                # RequestGuard middleware
│   ├── utils.py                   # Original MedRAG retrieval system
│   ├── medrag.py                  # Original MedRAG generator classes
│   └── template.py                # Prompt templates
├── templates/
│   └── index.html                 # Clean, professional web UI
├── static/
│   ├── css/style.css              # Minimal, accessible styling
│   └── js/main.js                 # Frontend logic
├── tests/
│   └── test_all.py                # Integration tests
├── scripts/
│   └── main.py                    # Evaluation CLI
├── .github/workflows/
│   └── ci.yml                     # CI: tests, security scan, lint
├── app.py                         # Flask entry point
├── wsgi.py                        # Gunicorn entry point
├── Dockerfile                     # Production container
├── requirements.txt               # Dependencies
├── .gitignore                     # Ignored files
├── .dockerignore                  # Docker ignored files
└── README.md                      # This file
```

---

## Installation

### Prerequisites

- Python 3.12+
- 8GB+ RAM (16GB recommended for full MedRAG mode)
- GPU optional (CPU works fine for inference; GPU accelerates embedding)

### Step 1: Clone & Environment

```bash
git clone https://github.com/Flamechargerr/MedRAG.git
cd MedRAG

python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: API Key (Required for LLM Generation)

Get a free Groq API key at [console.groq.com](https://console.groq.com):

```bash
export GROQ_API_KEY="gsk_..."
```

For production deployment, also set an auth token:

```bash
export APP_AUTH_TOKEN="your-secret-token-here"
```

### Step 3: Download Medical Corpora (Optional — for Full MedRAG Mode)

```bash
# Downloads automatically on first use, but you can pre-download:
python -c "
from src.utils import RetrievalSystem
rs = RetrievalSystem('MedCPT', 'Textbooks', './corpus')
print('Corpus ready')
"
```

This downloads ~2-5GB of medical text depending on the corpus. If you skip this, the system runs in **Fallback FAISS mode** with ~2K PubMed abstracts downloaded on-the-fly.

---

## Running the System

### Development Mode

```bash
export APP_ENV=local
export GROQ_API_KEY="your-key"
python app.py
```

Server starts on `http://localhost:5000`.

Open the web UI, set a corpus size (e.g., 200), click **Initialize**, then submit queries.

### Production Mode (Gunicorn)

```bash
export APP_ENV=production
export GROQ_API_KEY="your-key"
export APP_AUTH_TOKEN="your-secret-token"
export CORS_ORIGINS="https://yourdomain.com"

gunicorn \
  --bind 0.0.0.0:5000 \
  --workers 2 \
  --threads 4 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  wsgi:app
```

### Docker

```bash
docker build -t medrag .
docker run -p 5000:5000 \
  -e GROQ_API_KEY="your-key" \
  -e APP_AUTH_TOKEN="your-token" \
  -e APP_ENV=production \
  medrag
```

### Evaluation CLI

```bash
# Quick demo (2 questions, ~30 seconds)
python scripts/main.py --demo

# Full evaluation (100 questions, ~5-10 minutes)
python scripts/main.py

# Custom corpus
python scripts/main.py --corpus StatPearls --retriever MedCPT
```

---

## API Reference

### `POST /api/v1/init`

Initialize the retrieval engine. This downloads corpora and builds indices if not already present.

**Request:**
```json
{
  "corpus_size": 200,
  "force_reindex": false
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Loaded 1847 documents in 3.21s (full_medrag).",
  "mode": "full_medrag",
  "n_documents": 1847
}
```

### `POST /api/v1/chat`

Submit a medical question and receive a grounded, cited response.

**Request:**
```json
{
  "query": "What is the first-line treatment for type 2 diabetes?",
  "reference": "optional ground truth for evaluation"
}
```

**Response:**
```json
{
  "status": "success",
  "answer": "Metformin is the first-line pharmacological treatment for type 2 diabetes...",
  "baseline_answer": "The first-line treatment for type 2 diabetes is typically...",
  "sources": [
    {
      "id": "pubmed_12345",
      "title": "Metformin in the management of type 2 diabetes",
      "text": "Metformin reduces hepatic glucose production and improves insulin sensitivity...",
      "source": "pubmed",
      "score": 0.8234,
      "rank": 1
    }
  ],
  "metrics": {
    "latency": "2.34s",
    "latency_ms": 2340,
    "retrieval_confidence": 0.82,
    "n_sources": 5,
    "source_coverage": ["pubmed", "textbooks"],
    "citation_grounding": 0.78,
    "hallucination_risk": 0.22,
    "answer_confidence": 0.85,
    "rag_rouge_l": 0.62,
    "baseline_rouge_l": 0.41,
    "rouge_delta": 0.21
  }
}
```

### `GET /api/v1/health/live`

Liveness probe. Always returns `{"status": "ok"}` with HTTP 200.

### `GET /api/v1/health/ready`

Readiness probe. Returns HTTP 200 if initialized, HTTP 503 if still loading.

### `GET /api/v1/status`

Current engine status: initialization mode, corpus name, retriever name, document count, error state.

**Response:**
```json
{
  "is_initialized": true,
  "is_initializing": false,
  "error": null,
  "corpus_name": "Textbooks",
  "retriever_name": "MedCPT",
  "n_documents": 1847
}
```

---

## Metrics & Evaluation

### Metrics Explained

| Metric | What It Measures | Range | How to Interpret |
|--------|-----------------|-------|------------------|
| **retrieval_confidence** | Quality of the retrieved documents | 0.0 – 1.0 | Higher = better document relevance |
| **citation_grounding** | Fraction of answer sentences supported by evidence | 0.0 – 1.0 | Higher = more of the answer is traceable to sources |
| **hallucination_risk** | 1.0 − citation_grounding | 0.0 – 1.0 | Higher = more content not directly in retrieved docs |
| **answer_confidence** | Structural/lexical heuristic of LLM certainty | 0.0 – 1.0 | Higher = fewer hedging words, more citations |
| **rag_rouge_l** | ROUGE-L overlap with user-provided reference | 0.0 – 1.0 | Lexical similarity, NOT clinical accuracy |
| **baseline_rouge_l** | ROUGE-L of no-retrieval baseline | 0.0 – 1.0 | For comparison only |
| **rouge_delta** | Raw RAG − Baseline ROUGE-L | −1.0 – 1.0 | Positive = RAG has more lexical overlap |

### Important Caveats

- **ROUGE measures lexical overlap, not clinical accuracy.** A 0.15 difference in ROUGE-L does not mean 15% better patient outcomes.
- **Grounding is a proxy, not a fact-check.** An answer can be grounded but wrong if the retrieved evidence is outdated or incorrect.
- **Hallucination risk is retrieval-relative.** An answer may be factually correct but have high "hallucination risk" if the corpus is incomplete.
- **Confidence is a linguistic heuristic, not model uncertainty.** A confident-sounding wrong answer scores high.

### Evaluation Methodology

For systematic benchmarking, use the evaluation CLI:

```bash
python scripts/main.py --corpus Textbooks --retriever MedCPT
```

This runs the pipeline on the MedQA-USMLE evaluation set and reports aggregate statistics. Results are saved to `results/evaluation_results.json` with full per-question traces.

---

## Deployment

### Render (Free Tier)

1. Connect your GitHub repo to [Render](https://render.com)
2. Create a new **Web Service**
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn --bind 0.0.0.0:$PORT wsgi:app`
5. Add environment variables:
   - `GROQ_API_KEY`
   - `APP_AUTH_TOKEN`
   - `APP_ENV=production`
6. Deploy

### Railway

1. Connect repo to [Railway](https://railway.app)
2. Add variables in Settings → Variables
3. Deploy

### Self-Hosted (Docker)

```bash
docker build -t medrag .
docker run -d \
  -p 5000:5000 \
  -e GROQ_API_KEY="your-key" \
  -e APP_AUTH_TOKEN="your-token" \
  -e APP_ENV=production \
  --name medrag \
  medrag
```

### Health Monitoring

Use the readiness/liveness probes for Kubernetes or Docker Compose:

```yaml
livenessProbe:
  httpGet:
    path: /api/v1/health/live
    port: 5000
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /api/v1/health/ready
    port: 5000
  initialDelaySeconds: 5
  periodSeconds: 10
```

---

## Testing

```bash
# Run all tests
python -m unittest tests.test_all -v

# Run specific test suite
python -m unittest tests.test_all.TestAppAPI -v
python -m unittest tests.test_all.TestSecurity -v
python -m unittest tests.test_all.TestMetrics -v
```

Tests use real dependency imports and verify actual routing, security, and metric logic. No mocks of core behavior.

---

## Environment Variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `APP_ENV` | `local` | No | `local`, `staging`, or `production` |
| `GROQ_API_KEY` | — | **Yes** | Groq API key for LLM generation |
| `APP_AUTH_TOKEN` | — | In prod | Bearer token for API auth |
| `CORS_ORIGINS` | `http://localhost:5000` | No | Comma-separated allowed origins |
| `RATE_LIMIT_PER_MINUTE` | `60` | No | Per-IP+path rate limit |
| `MAX_REQUEST_BYTES` | `65536` | No | Max request body size (bytes) |
| `MAX_QUERY_CHARS` | `2000` | No | Max query length |
| `CORPUS_NAME` | `Textbooks` | No | MedRAG corpus name |
| `RETRIEVER_NAME` | `MedCPT` | No | MedRAG retriever name |
| `FAISS_DB_DIR` | `./faiss_db` | No | Fallback vector store directory |
| `LOG_LEVEL` | `INFO` | No | Logging verbosity |
| `LLM_MAX_RETRIES` | `2` | No | LLM retry count on failure |
| `DEFAULT_CORPUS_SIZE` | `200` | No | Default corpus size for init |
| `MAX_CORPUS_SIZE` | `2000` | No | Max allowed corpus size |
| `RETRIEVAL_TOP_K` | `5` | No | Number of documents to retrieve |

---

## Troubleshooting

### "Engine initialization failed" / fallback mode

The full MedRAG mode requires downloading corpora from HuggingFace (~2-5GB). If you see:
```
Loaded 200 documents in 1.2s (fallback_faiss).
```

This is normal. The system is running in fallback mode with PubMedBERT embeddings on a smaller corpus. To enable full mode, run:
```bash
python -c "from src.utils import RetrievalSystem; rs = RetrievalSystem('MedCPT', 'Textbooks', './corpus')"
```

### High latency (>5s)

- **CPU mode:** First embedding download takes ~30s. Subsequent queries are faster.
- **GPU mode:** Ensure `torch.cuda.is_available()` is true.
- **Rate limits:** Groq free tier has rate limits. Set `LLM_MAX_RETRIES=3`.

### "Unauthorized" errors

In production mode (`APP_ENV=production`), all API endpoints require:
```
Authorization: Bearer <APP_AUTH_TOKEN>
```

### NLTK / numpy compatibility errors

If you see `numpy.dtype size changed` errors, your sklearn/numpy versions are incompatible. Fix with:
```bash
pip install --upgrade numpy scikit-learn rouge-score
```

---

## License

MIT License — see [LICENSE](LICENSE).

## Citation

If you use this system in research, please cite the original MedRAG paper:

```bibtex
@article{xiong2024medrag,
  title={Benchmarking Retrieval-Augmented Generation for Medicine},
  author={Xiong, Guangzhi and Jin, Qiyao and Zhu, Shichao and Jin, Qin and Zhou, Zhiyuan and Hou, Xun and Jin, Qiaozhu and Chen, Zhengliang and Lu, Xiang and Chen, Sheng and others},
  journal={arXiv preprint arXiv:2402.13178},
  year={2024}
}
```

## Acknowledgments

- Original MedRAG implementation: [RUCAIBox/MedRAG](https://github.com/RUCAIBox/MedRAG)
- MedCPT models: [NCBI](https://github.com/ncbi/MedCPT)
- Groq API: [console.groq.com](https://console.groq.com)
