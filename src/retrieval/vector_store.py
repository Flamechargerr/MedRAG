import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(self, db_dir="./chroma_db", embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
        logger.info(f"Initializing ChromaDB Vector Store at {db_dir}")
        
        # Use the modern ChromaDB initialization
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        
        logger.info(f"Loading embedding model {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.device = device
        
        # Get or create collection
        self.collection_name = "medical_corpus"
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection with {self.collection.count()} documents")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Medical corpus for RAG retrieval"}
            )
            logger.info("Created new ChromaDB collection")

    def encode_corpus(self, corpus_texts):
        logger.info("Encoding entire corpus...")
        corpus_embeddings = self.embedding_model.encode(
            corpus_texts, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device
        )
        return corpus_embeddings.cpu().numpy()

    def add_documents(self, documents, embeddings=None):
        if not documents:
            return
            
        logger.info(f"Adding {len(documents)} documents to ChromaDB...")
        if embeddings is None:
            texts = [doc['text'] for doc in documents]
            embeddings = self.encode_corpus(texts)
            
        for i, doc in enumerate(tqdm(documents)):
            # Handle duplicates gracefully or skip existing
            try:
                self.collection.add(
                    embeddings=[embeddings[i].tolist()],
                    documents=[doc['text']],
                    ids=[doc.get('id', str(i))],
                    metadatas=[{"title": doc.get('title', 'Unknown')}]
                )
            except Exception as e:
                pass # Document likely exists
        logger.info(f"Collection now has {self.collection.count()} documents")
        return embeddings

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True, device=self.device)
        query_embedding = query_embedding.cpu().numpy()
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        retrieved_docs = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        return retrieved_docs
