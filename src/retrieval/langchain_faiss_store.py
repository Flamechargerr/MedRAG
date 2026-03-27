import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm.auto import tqdm
import os

logger = logging.getLogger(__name__)

class LangchainFAISSStore:
    def __init__(self, db_dir="./faiss_db", embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
        logger.info(f"Initializing LangChain FAISS Vector Store at {db_dir}")
        self.db_dir = db_dir
        
        logger.info(f"Loading PyTorch embedding model: {embedding_model_name} on {device}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        self.vector_store = None
        self._load_or_init_store()
        
    def _load_or_init_store(self):
        if os.path.exists(self.db_dir) and os.path.exists(os.path.join(self.db_dir, "index.faiss")):
            logger.info("Loading existing FAISS index from disk...")
            try:
                self.vector_store = FAISS.load_local(self.db_dir, self.embeddings, allow_dangerous_deserialization=True)
                logger.info(f"Loaded FAISS index with {self.vector_store.index.ntotal} documents.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.vector_store = None
        else:
            logger.info("No existing FAISS index found. Ready to create a new one upon adding documents.")

    def add_documents(self, documents: List[Dict[str, Any]]):
        if not documents:
            return
            
        logger.info(f"Adding {len(documents)} documents to FAISS via LangChain...")
        
        texts = [doc['text'] for doc in documents]
        metadatas = [{"title": doc.get('title', 'Unknown'), "id": doc.get('id', str(i))} 
                     for i, doc in enumerate(documents)]
                     
        if self.vector_store is None:
            # Initialize from texts
            self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            # Add to existing
            self.vector_store.add_texts(texts, metadatas=metadatas)
            
        logger.info(f"Collection now has {self.vector_store.index.ntotal} documents.")
        
        # Save to disk
        os.makedirs(self.db_dir, exist_ok=True)
        self.vector_store.save_local(self.db_dir)
        logger.info(f"Saved FAISS index to {self.db_dir}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.vector_store is None:
            logger.warning("Vector store is not initialized. Retrieve called before adding documents.")
            return []
            
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        retrieved_docs = []
        for doc, score in docs_with_scores:
            retrieved_docs.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'distance': float(score)  # L2 distance
            })
            
        return retrieved_docs
