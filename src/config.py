import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# API Keys
OPENCALL_LLM_KEY = (
    os.environ.get("GROQ_API_KEY")
    or os.environ.get("OPENCALL_LLM_KEY")
    or os.environ.get("EMERGENT_LLM_KEY")
    or ""
)
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', '')

# Model Configurations
class Config:
    DEVICE = 'cuda' if False else 'cpu' # Auto-detect in main but set a fallback
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EMBEDDING_DIM = 384
    LSTM_HIDDEN_DIM = 256
    LSTM_NUM_LAYERS = 2
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    TOP_K_RETRIEVAL = 5
    NUM_EVAL_QUESTIONS = 100
    
    # Paths
    CHROMA_DB_DIR = "./chroma_db"
    MODEL_SAVE_PATH = "./lstm_retriever_model.pt"

    @classmethod
    def load_env(cls):
        # Allow checking GPU availability correctly at runtime
        try:
            import torch
            cls.DEVICE = torch.device(
                'cuda' if torch.cuda.is_available()
                else 'mps' if torch.backends.mps.is_available()
                else 'cpu'
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to initialize PyTorch device, falling back to CPU: {e}"
            )
            cls.DEVICE = 'cpu'
