import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass(frozen=True)
class RuntimeConfig:
    environment: str
    log_level: str
    api_key: str
    huggingface_api_key: str
    app_auth_token: str
    cors_origins: List[str]
    rate_limit_per_minute: int
    max_request_bytes: int
    max_query_chars: int
    default_corpus_size: int
    max_corpus_size: int
    retrieval_top_k: int
    faiss_db_dir: str
    runtime_state_path: str
    llm_max_retries: int
    corpus_name: str
    retriever_name: str
    device: str

    @property
    def require_auth(self) -> bool:
        return self.environment in {"staging", "production"}


def _parse_cors_origins(raw_value: str) -> List[str]:
    if not raw_value:
        return ["http://127.0.0.1:5000", "http://localhost:5000"]
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _load_json_secrets(path: Optional[str]) -> dict:
    if not path:
        return {}
    secret_path = Path(path)
    if not secret_path.exists():
        raise ValueError(f"Secret file not found: {secret_path}")
    with secret_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Secret file payload must be a JSON object")
    return payload


def _read_api_key(secret_payload: dict) -> str:
    explicit = os.environ.get("GROQ_API_KEY", "")
    if explicit:
        return explicit
    secret_file_key = str(secret_payload.get("GROQ_API_KEY", "")).strip()
    if secret_file_key:
        return secret_file_key

    legacy = (
        os.environ.get("OPENCALL_LLM_KEY", "")
        or os.environ.get("EMERGENT_LLM_KEY", "")
    )
    if legacy:
        logger.warning("Using legacy API key env var; set GROQ_API_KEY for production use.")
    return legacy


@lru_cache(maxsize=1)
def get_config() -> RuntimeConfig:
    environment = os.environ.get("APP_ENV", "local").strip().lower()
    if environment not in {"local", "staging", "production"}:
        raise ValueError("APP_ENV must be one of: local, staging, production")

    secret_payload = _load_json_secrets(os.environ.get("APP_SECRET_FILE"))
    api_key = _read_api_key(secret_payload)

    app_auth_token = os.environ.get("APP_AUTH_TOKEN", "") or str(secret_payload.get("APP_AUTH_TOKEN", "")).strip()
    cors_origins = _parse_cors_origins(os.environ.get("CORS_ORIGINS", ""))

    # Device detection
    device = "cpu"
    try:
        import torch
        device = str(torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        ))
    except Exception:
        pass

    config = RuntimeConfig(
        environment=environment,
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        api_key=api_key,
        huggingface_api_key=os.environ.get("HUGGINGFACE_API_KEY", ""),
        app_auth_token=app_auth_token,
        cors_origins=cors_origins,
        rate_limit_per_minute=int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60")),
        max_request_bytes=int(os.environ.get("MAX_REQUEST_BYTES", str(64 * 1024))),
        max_query_chars=int(os.environ.get("MAX_QUERY_CHARS", "2000")),
        default_corpus_size=int(os.environ.get("DEFAULT_CORPUS_SIZE", "200")),
        max_corpus_size=int(os.environ.get("MAX_CORPUS_SIZE", "2000")),
        retrieval_top_k=int(os.environ.get("RETRIEVAL_TOP_K", "5")),
        faiss_db_dir=os.environ.get("FAISS_DB_DIR", "./faiss_db"),
        runtime_state_path=os.environ.get("RUNTIME_STATE_PATH", "./runtime_state.json"),
        llm_max_retries=int(os.environ.get("LLM_MAX_RETRIES", "2")),
        corpus_name=os.environ.get("CORPUS_NAME", "Textbooks"),
        retriever_name=os.environ.get("RETRIEVER_NAME", "MedCPT"),
        device=device,
    )

    if config.require_auth and not config.app_auth_token:
        raise ValueError("APP_AUTH_TOKEN is required in staging/production")
    if config.require_auth and "*" in config.cors_origins:
        raise ValueError("Wildcard CORS is not allowed in staging/production")

    return config


# Backward compatibility exports
OPENCALL_LLM_KEY = get_config().api_key
HUGGINGFACE_API_KEY = get_config().huggingface_api_key


class Config:
    DEVICE = get_config().device
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EMBEDDING_DIM = 384
    LSTM_HIDDEN_DIM = 256
    LSTM_NUM_LAYERS = 2
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    TOP_K_RETRIEVAL = get_config().retrieval_top_k
    NUM_EVAL_QUESTIONS = 100

    CHROMA_DB_DIR = "./chroma_db"
    MODEL_SAVE_PATH = "./lstm_retriever_model.pt"

    @classmethod
    def load_env(cls):
        try:
            import torch
            cls.DEVICE = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        except Exception as exc:
            logging.getLogger(__name__).warning("Failed to initialize PyTorch device, falling back to CPU: %s", exc)
            cls.DEVICE = "cpu"
# feat: add async indexing support for large corpora
# feat: implement streaming response for chat endpoint
