from flask import Flask, request, jsonify, render_template
import logging
import time
from src.config import OPENCALL_LLM_KEY, Config
from src.data_loader import load_medqa_data, load_medical_corpus
from src.retrieval.langchain_faiss_store import LangchainFAISSStore
from src.generation.llm_generators import GroqGenerator
from src.evaluation.metrics import RAGEvaluator

app = Flask(__name__, static_folder='static', template_folder='templates')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedBot-Flask")

class MedBotState:
    def __init__(self):
        self.vector_store = None
        self.llm_gen = None
        self.evaluator = RAGEvaluator()
        self.is_initialized = False
        self.current_reference = ""

system = MedBotState()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/init", methods=["POST"])
def initialize():
    if system.is_initialized:
        return jsonify({"status": "success", "message": "System already initialized."})
    
    try:
        data = request.json or {}
        corpus_size = data.get("corpus_size", 200)
        
        start_time = time.time()
        Config.load_env()
        logger.info(f"Initializing Medical Knowledge Base (Size: {corpus_size})...")
        
        _, full_dataset = load_medqa_data(num_eval_questions=5)
        medical_corpus = load_medical_corpus(dataset_to_fallback=full_dataset, max_docs=corpus_size)
        
        system.vector_store = LangchainFAISSStore(db_dir="./faiss_db", device=Config.DEVICE)
        system.vector_store.add_documents(medical_corpus)
        
        system.llm_gen = GroqGenerator(OPENCALL_LLM_KEY)
        system.is_initialized = True
        
        duration = time.time() - start_time
        return jsonify({"status": "success", "message": f"Loaded {len(medical_corpus)} docs in {duration:.2f}s."})
    except Exception as e:
        logger.error(f"Init error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    if not system.is_initialized:
        return jsonify({"status": "error", "message": "Initialize system first"}), 400
        
    data = request.json
    question = data.get("query", "")
    reference = data.get("reference", "")
    system.current_reference = reference
    
    if not question:
        return jsonify({"status": "error", "message": "No query provided"}), 400
        
    start_time = time.time()
    try:
        retrieved_docs = system.vector_store.retrieve(question, top_k=3)
        answer = system.llm_gen.generate(question, retrieved_docs)
        baseline_answer = system.llm_gen.generate_no_context(question)
        
        latency = time.time() - start_time
        
        metrics = {"Latency": f"{latency:.2f}s"}
        if system.current_reference:
            rag_scores = system.evaluator.compute_rouge_scores(answer, system.current_reference)
            base_scores = system.evaluator.compute_rouge_scores(baseline_answer, system.current_reference)
            metrics["RAG_ROUGE_L"] = round(rag_scores["rougeL"], 4)
            metrics["Baseline_ROUGE_L"] = round(base_scores["rougeL"], 4)
            val_diff = (rag_scores['rougeL'] - base_scores['rougeL'])
            metrics["Accuracy_Improvement"] = f"{val_diff * 100:.1f}%"
            
        sources = [{"title": d["metadata"].get("title", "Source"), "text": d["text"][:400] + "..."} for d in retrieved_docs]
        
        return jsonify({
            "status": "success",
            "answer": answer,
            "baseline_answer": baseline_answer,
            "sources": sources,
            "metrics": metrics
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
