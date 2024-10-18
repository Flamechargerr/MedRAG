"""Data loading with real medical corpora and honest fallbacks.

- Loads PubMed abstracts via HuggingFace datasets when available.
- Loads StatPearls, Textbooks, and Wikipedia from the MedRAG corpus repos.
- Falls back to MedQA questions ONLY as a last resort, clearly labeled.
- All corpora are filtered to ensure non-empty text content.
"""

import logging
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Corpus loaders
# ------------------------------------------------------------------

def load_medqa_data(num_eval_questions: int = 100, split: str = "test"):
    """Load MedQA USMLE or MedMCQA as a fallback evaluation set."""
    logger.info("Loading evaluation dataset...")
    try:
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split, trust_remote_code=True)
        logger.info("Loaded %d questions from MedQA-USMLE", len(dataset))
    except Exception as e:
        logger.warning("MedQA load failed: %s. Trying MedMCQA fallback.", e)
        try:
            dataset = load_dataset("medmcqa", split="validation", trust_remote_code=True)
            logger.info("Loaded %d questions from MedMCQA", len(dataset))
        except Exception as e2:
            logger.error("All evaluation datasets failed: %s", e2)
            raise RuntimeError(f"Could not load evaluation data: {e2}") from e2

    eval_questions = dataset.shuffle(seed=42).select(
        range(min(num_eval_questions, len(dataset)))
    )
    return eval_questions, dataset


def load_pubmed_corpus(max_docs: int = 5000) -> List[Dict[str, Any]]:
    """Load PubMed abstracts via the PubMed dataset on HuggingFace.

    The correct dataset is 'pubmed' which provides MedlineCitation records.
    """
    logger.info("Loading PubMed corpus (max %d docs)...", max_docs)
    try:
        # pubmed dataset is large; sample a slice
        dataset = load_dataset("pubmed", split="train", streaming=True, trust_remote_code=True)
        docs = []
        for i, item in enumerate(dataset):
            if i >= max_docs:
                break
            text = _extract_pubmed_text(item)
            if text:
                docs.append({
                    "text": text,
                    "id": f"pubmed_{i}",
                    "title": item.get("MedlineCitation", {}).get("Article", {}).get("ArticleTitle", "Untitled"),
                })
        logger.info("Loaded %d PubMed documents", len(docs))
        return docs
    except Exception as e:
        logger.warning("PubMed corpus loading failed: %s", e)
        return []


def load_medical_corpus(
    dataset_to_fallback=None, max_docs: int = 5000, corpus_sources: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Load a medical corpus from multiple sources.

    Priority order:
    1. PubMed abstracts (if available)
    2. User-specified corpus_sources (e.g., ['textbooks', 'statpearls'])
    3. MedQA question fallback (clearly labeled, last resort)
    """
    corpus: List[Dict[str, Any]] = []

    # 1. Try PubMed
    pubmed_docs = load_pubmed_corpus(max_docs=max_docs)
    corpus.extend(pubmed_docs)

    # 2. Try user-specified corpora
    if corpus_sources:
        for source in corpus_sources:
            try:
                source_docs = _load_corpus_source(source, max_docs=max_docs)
                corpus.extend(source_docs)
                logger.info("Loaded %d documents from %s", len(source_docs), source)
            except Exception as e:
                logger.warning("Failed to load corpus source '%s': %s", source, e)

    # 3. Fallback to MedQA questions (only if absolutely nothing else loaded)
    if not corpus and dataset_to_fallback is not None:
        logger.warning("No real corpus loaded. Using MedQA questions as fallback corpus.")
        corpus = [
            {
                "text": item.get("question", item.get("sent1", "")),
                "id": f"fallback_{i}",
                "title": f"Medical Question {i}",
                "source": "fallback_medqa",
            }
            for i, item in enumerate(dataset_to_fallback)
            if item.get("question") or item.get("sent1")
        ]

    if not corpus:
        raise RuntimeError("Could not load any medical corpus. Check network connectivity and dataset availability.")

    # Deduplicate and filter empty
    seen = set()
    filtered = []
    for doc in corpus:
        text = doc.get("text", "").strip()
        if not text or len(text) < 20:
            continue
        key = text[:100]
        if key in seen:
            continue
        seen.add(key)
        filtered.append(doc)

    logger.info("Medical corpus prepared: %d unique, non-empty documents.", len(filtered))
    return filtered[:max_docs]


# ------------------------------------------------------------------
# Internal corpus source loaders
# ------------------------------------------------------------------

def _load_corpus_source(source: str, max_docs: int = 5000) -> List[Dict[str, Any]]:
    """Load a specific corpus source.  These are the MedRAG HF datasets."""
    if source == "textbooks":
        return _load_hf_jsonl_corpus("MedRAG/textbooks", max_docs)
    elif source == "statpearls":
        return _load_hf_jsonl_corpus("MedRAG/statpearls", max_docs)
    elif source == "wikipedia":
        return _load_hf_jsonl_corpus("MedRAG/wikipedia", max_docs)
    elif source == "pubmed":
        return load_pubmed_corpus(max_docs)
    else:
        logger.warning("Unknown corpus source: %s", source)
        return []


def _load_hf_jsonl_corpus(dataset_name: str, max_docs: int) -> List[Dict[str, Any]]:
    """Load a MedRAG corpus from HuggingFace (JSONL format)."""
    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
        docs = []
        for i, item in enumerate(dataset):
            if i >= max_docs:
                break
            text = item.get("content", item.get("text", ""))
            title = item.get("title", "Untitled")
            if text and len(text.strip()) > 20:
                docs.append({
                    "text": text.strip(),
                    "id": item.get("id", f"{dataset_name}_{i}"),
                    "title": title,
                    "source": dataset_name.split("/")[-1],
                })
        return docs
    except Exception as e:
        logger.warning("Failed to load HF dataset %s: %s", dataset_name, e)
        return []


def _extract_pubmed_text(item: Dict[str, Any]) -> str:
    """Extract abstract text from a PubMed MedlineCitation record."""
    try:
        article = item.get("MedlineCitation", {}).get("Article", {})
        abstract = article.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract, list):
            return " ".join(str(a) for a in abstract if a)
        return str(abstract)
    except Exception:
        return ""


# ------------------------------------------------------------------
# PyTorch Dataset wrapper
# ------------------------------------------------------------------

class MedicalQADataset(Dataset):
    """PyTorch Dataset wrapper for LSTM training."""

    def __init__(self, questions, tokenizer, max_length: int = 128):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        if isinstance(question, dict):
            text = question.get("question", question.get("sent1", ""))
        else:
            text = str(question)

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "text": text,
        }
