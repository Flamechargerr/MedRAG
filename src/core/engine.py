"""MedRAG Core Engine — Production-ready wrapper around the authentic MedRAG retrieval system.

This module bridges the original MedRAG research code (src/utils.py) with the web service
layer, providing async-safe initialization, proper caching, and graceful degradation.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedDocument:
    """A single retrieved document with provenance metadata."""

    id: str
    title: str
    text: str
    source: str  # e.g. 'pubmed', 'textbooks', 'statpearls', 'wikipedia'
    score: float
    rank: int


@dataclass(frozen=True)
class GenerationResult:
    """Result of a single generation pass."""

    answer: str
    sources: List[RetrievedDocument]
    raw_context: str
    latency_ms: int
    retrieval_confidence: float


@dataclass
class EngineState:
    """Serializable runtime state for persistence."""

    is_initialized: bool = False
    initialized_at: Optional[int] = None
    corpus_name: str = ""
    retriever_name: str = ""
    n_documents: int = 0
    duration_seconds: float = 0.0


class MedRAGEngine:
    """Production wrapper around the authentic MedRAG retrieval system.

    Design goals:
    1. Use the real MedRAG retriever (MedCPT + multi-corpus + RRF) when corpora exist.
    2. Gracefully fall back to lightweight LangChain FAISS when full corpora are absent.
    3. Thread-safe, async-safe initialization with background indexing support.
    4. Proper citation tracking and retrieval confidence scoring.
    """

    def __init__(
        self,
        db_dir: str = "./corpus",
        cache_dir: Optional[str] = None,
        corpus_name: str = "Textbooks",
        retriever_name: str = "MedCPT",
        faiss_fallback_dir: str = "./faiss_db",
        device: str = "cpu",
        state_path: str = "./runtime_state.json",
        retrieval_top_k: int = 5,
    ):
        self.db_dir = Path(db_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.corpus_name = corpus_name
        self.retriever_name = retriever_name
        self.faiss_fallback_dir = faiss_fallback_dir
        self.device = device
        self.state_path = Path(state_path)
        self.retrieval_top_k = retrieval_top_k

        self._lock = threading.RLock()
        self._init_lock = threading.Lock()
        self._is_initialized = False
        self._init_error: Optional[str] = None
        self._init_thread: Optional[threading.Thread] = None

        # Core MedRAG components (lazy-loaded)
        self._retrieval_system: Optional[Any] = None
        self._fallback_store: Optional[Any] = None
        self._doc_extractor: Optional[Any] = None

        # Caches
        self._retrieval_cache: OrderedDict[str, List[RetrievedDocument]] = OrderedDict()
        self._retrieval_cache_size = 128

        # Load persisted state
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _load_state(self) -> EngineState:
        if not self.state_path.exists():
            return EngineState()
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return EngineState(
                is_initialized=payload.get("is_initialized", False),
                initialized_at=payload.get("initialized_at"),
                corpus_name=payload.get("corpus_name", ""),
                retriever_name=payload.get("retriever_name", ""),
                n_documents=payload.get("n_documents", 0),
                duration_seconds=payload.get("duration_seconds", 0.0),
            )
        except Exception as exc:
            logger.warning("Failed to load engine state: %s", exc)
            return EngineState()

    def _save_state(self) -> None:
        try:
            with self.state_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "is_initialized": self._state.is_initialized,
                        "initialized_at": self._state.initialized_at,
                        "corpus_name": self._state.corpus_name,
                        "retriever_name": self._state.retriever_name,
                        "n_documents": self._state.n_documents,
                        "duration_seconds": self._state.duration_seconds,
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:
            logger.warning("Failed to save engine state: %s", exc)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def health_ready(self) -> bool:
        with self._lock:
            return self._is_initialized and self._init_error is None

    def health_live(self) -> bool:
        return True  # Process is alive

    def get_init_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "is_initialized": self._is_initialized,
                "is_initializing": self._init_thread is not None and self._init_thread.is_alive(),
                "error": self._init_error,
                "corpus_name": self._state.corpus_name,
                "retriever_name": self._state.retriever_name,
                "n_documents": self._state.n_documents,
            }

    def initialize_blocking(self) -> Dict[str, Any]:
        """Synchronous initialization for CLI / testing."""
        with self._init_lock:
            if self._is_initialized:
                return {"status": "success", "message": "System already initialized."}
            return self._do_initialize()

    def initialize_background(self) -> Dict[str, Any]:
        """Non-blocking initialization for the web server."""
        with self._init_lock:
            if self._is_initialized:
                return {"status": "success", "message": "System already initialized."}
            if self._init_thread is not None and self._init_thread.is_alive():
                return {"status": "pending", "message": "Initialization already in progress."}

            self._init_thread = threading.Thread(target=self._do_initialize, daemon=True)
            self._init_thread.start()
            return {"status": "pending", "message": "Initialization started in background."}

    def _do_initialize(self) -> Dict[str, Any]:
        start_time = time.time()
        try:
            self._init_error = None
            self._try_setup_medrag_retrieval()
            if self._retrieval_system is not None:
                mode = "full_medrag"
            else:
                self._setup_fallback_retrieval()
                mode = "fallback_faiss"

            duration = time.time() - start_time
            n_docs = self._count_documents()

            with self._lock:
                self._is_initialized = True
                self._state.is_initialized = True
                self._state.initialized_at = int(time.time())
                self._state.corpus_name = self.corpus_name
                self._state.retriever_name = self.retriever_name
                self._state.n_documents = n_docs
                self._state.duration_seconds = round(duration, 2)
                self._save_state()

            logger.info(
                "MedRAG engine initialized in %.2fs (%s mode, %d docs).",
                duration,
                mode,
                n_docs,
            )
            return {
                "status": "success",
                "message": f"Loaded {n_docs} documents in {duration:.2f}s ({mode}).",
                "mode": mode,
                "n_documents": n_docs,
            }
        except Exception as exc:
            logger.exception("Engine initialization failed")
            with self._lock:
                self._init_error = str(exc)
            return {"status": "error", "message": str(exc)}

    # ------------------------------------------------------------------
    # Retrieval backends
    # ------------------------------------------------------------------
    def _try_setup_medrag_retrieval(self) -> None:
        """Attempt to load the authentic MedRAG retrieval system."""
        try:
            # Import here to avoid heavy imports at module load time
            from src.utils import RetrievalSystem

            logger.info(
                "Attempting to load MedRAG retrieval system: %s / %s",
                self.retriever_name,
                self.corpus_name,
            )
            self._retrieval_system = RetrievalSystem(
                retriever_name=self.retriever_name,
                corpus_name=self.corpus_name,
                db_dir=str(self.db_dir),
                HNSW=False,
                cache=False,  # we handle caching ourselves
            )
            logger.info("MedRAG retrieval system loaded successfully.")
        except Exception as exc:
            logger.warning(
                "Could not load full MedRAG retrieval system (%s). "
                "Will attempt fallback. Error: %s",
                type(exc).__name__,
                exc,
            )
            self._retrieval_system = None

    def _setup_fallback_retrieval(self) -> None:
        """Set up lightweight LangChain FAISS fallback with a medical embedding model."""
        from src.retrieval.langchain_faiss_store import LangchainFAISSStore
        from src.data_loader import load_medical_corpus, load_medqa_data

        logger.info("Setting up fallback FAISS retrieval...")

        self._fallback_store = LangchainFAISSStore(
            db_dir=self.faiss_fallback_dir,
            embedding_model_name="NeuML/pubmedbert-base-embeddings",  # Medical-specific
            device=self.device,
        )

        has_existing = (
            self._fallback_store.vector_store is not None
            and getattr(self._fallback_store.vector_store.index, "ntotal", 0) > 0
        )
        if not has_existing:
            logger.info("Building fallback corpus from PubMed abstracts...")
            try:
                _, full_dataset = load_medqa_data(num_eval_questions=5)
                corpus = load_medical_corpus(
                    dataset_to_fallback=full_dataset, max_docs=2000
                )
                self._fallback_store.add_documents(corpus)
            except Exception as exc:
                logger.warning("Fallback corpus build failed: %s", exc)
                raise

    def _count_documents(self) -> int:
        if self._retrieval_system is not None:
            # Estimate: sum of retriever corpus sizes
            try:
                return sum(
                    r.index.ntotal
                    for retriever_list in self._retrieval_system.retrievers
                    for r in retriever_list
                    if hasattr(r, "index") and hasattr(r.index, "ntotal")
                )
            except Exception:
                return 0
        elif self._fallback_store is not None and self._fallback_store.vector_store is not None:
            return getattr(self._fallback_store.vector_store.index, "ntotal", 0)
        return 0

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------
    def _cache_key(self, query: str, top_k: int) -> str:
        return f"{query.strip().lower()}:{top_k}"

    def _cache_get(self, key: str) -> Optional[List[RetrievedDocument]]:
        if key not in self._retrieval_cache:
            return None
        self._retrieval_cache.move_to_end(key)
        return self._retrieval_cache[key]

    def _cache_set(self, key: str, value: List[RetrievedDocument]) -> None:
        self._retrieval_cache[key] = value
        self._retrieval_cache.move_to_end(key)
        if len(self._retrieval_cache) > self._retrieval_cache_size:
            self._retrieval_cache.popitem(last=False)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        if not self._is_initialized:
            raise RuntimeError("Engine is not initialized")

        k = top_k or self.retrieval_top_k
        cache_key = self._cache_key(query, k)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if self._retrieval_system is not None:
            docs = self._retrieve_medrag(query, k)
        else:
            docs = self._retrieve_fallback(query, k)

        self._cache_set(cache_key, docs)
        return docs

    def _retrieve_medrag(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """Use the authentic MedRAG retrieval system."""
        texts, scores = self._retrieval_system.retrieve(
            question=query, k=top_k, rrf_k=100, id_only=False
        )
        results = []
        for rank, (doc, score) in enumerate(zip(texts, scores), start=1):
            results.append(
                RetrievedDocument(
                    id=doc.get("id", f"doc_{rank}"),
                    title=doc.get("title", "Untitled"),
                    text=doc.get("content", doc.get("text", "")),
                    source=doc.get("source", self.corpus_name.lower()),
                    score=float(score),
                    rank=rank,
                )
            )
        return results

    def _retrieve_fallback(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """Use the lightweight FAISS fallback."""
        raw_docs = self._fallback_store.retrieve(query, top_k=top_k)
        results = []
        for rank, doc in enumerate(raw_docs, start=1):
            # Convert distance to a rough confidence score (smaller distance = higher score)
            distance = doc.get("distance", 0.0)
            confidence = max(0.0, 1.0 - distance / 10.0)  # heuristic
            results.append(
                RetrievedDocument(
                    id=doc.get("metadata", {}).get("id", f"doc_{rank}"),
                    title=doc.get("metadata", {}).get("title", "Source"),
                    text=doc.get("text", ""),
                    source="fallback_faiss",
                    score=confidence,
                    rank=rank,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Confidence / utility
    # ------------------------------------------------------------------
    def compute_retrieval_confidence(self, docs: List[RetrievedDocument]) -> float:
        """Compute a normalized confidence score from retrieval results."""
        if not docs:
            return 0.0
        # For MedRAG, scores are RRF scores (higher = better)
        # For fallback, scores are already normalized [0,1]
        if docs[0].source == "fallback_faiss":
            return round(sum(d.score for d in docs) / len(docs), 3)
        # RRF scores: use top score as proxy
        top_score = docs[0].score
        return round(min(1.0, top_score / 0.5), 3)  # 0.5 is a typical good RRF score

    def clear_cache(self) -> None:
        with self._lock:
            self._retrieval_cache.clear()
