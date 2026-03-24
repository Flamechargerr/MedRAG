import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import gradio as gr
import logging
import time
from src.config import OPENCALL_LLM_KEY, Config
from src.data_loader import load_medqa_data, load_medical_corpus
from src.retrieval.vector_store import ChromaVectorStore
from src.generation.llm_generators import GroqGenerator
from src.evaluation.metrics import RAGEvaluator

# Configure elegant logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedBot-Premium")

# --- Global System State ---
class MedBotState:
    def __init__(self):
        self.vector_store: ChromaVectorStore | None = None
        self.llm_gen: GroqGenerator | None = None
        self.evaluator: RAGEvaluator = RAGEvaluator()
        self.is_initialized: bool = False
        self.current_reference: str = ""
        self.stats = {"last_latency": 0.0, "total_queries": 0}

system = MedBotState()

# --- Backend Logic ---
def initialize_system(corpus_size: int = 200):
    if system.is_initialized:
        return "⚡ System already optimized and ready."
    
    try:
        start_time = time.time()
        Config.load_env()
        logger.info(f"Initializing Medical Knowledge Base (Size: {corpus_size})...")
        
        # Load sample data for UI demo
        _, full_dataset = load_medqa_data(num_eval_questions=5)
        medical_corpus = load_medical_corpus(dataset_to_fallback=full_dataset, max_docs=corpus_size)
        
        # Ensure imports are resolved correctly in the environment
        system.vector_store = ChromaVectorStore(db_dir=Config.CHROMA_DB_DIR, device=Config.DEVICE)
        system.vector_store.add_documents(medical_corpus)
        
        system.llm_gen = GroqGenerator(OPENCALL_LLM_KEY)
        system.is_initialized = True
        
        duration = time.time() - start_time
        return f"✅ Knowledge Base Loaded ({len(medical_corpus)} docs) in {duration:.2f}s. Ready for queries."
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return f"❌ System Error: {str(e)}"

def process_query(question: str):
    if not system.is_initialized:
        return "⚠️ Please initialize the Knowledge Base first.", "", "N/A", None
    
    if not question.strip():
        return "⚠️ Please provide a medical question.", "", "N/A", None
    
    # Assertions to satisfy type checker
    assert system.vector_store is not None, "Vector store not initialized"
    assert system.llm_gen is not None, "LLM generator not initialized"
    
    start_time = time.time()
    try:
        # 1. Retrieval
        retrieved_docs = system.vector_store.retrieve(question, top_k=3)
        
        # 2. Generation
        answer = system.llm_gen.generate(question, retrieved_docs)
        
        # 2b. Baseline Generation (No Context)
        baseline_answer = system.llm_gen.generate_no_context(question)
        
        latency = time.time() - start_time
        system.stats["last_latency"] = latency
        system.stats["total_queries"] += 1
        
        # 3. Metrics (using stored reference if available)
        metrics_json = {"Latency": f"{latency:.2f}s"}
        if system.current_reference:
            rag_scores = system.evaluator.compute_rouge_scores(answer, system.current_reference)
            base_scores = system.evaluator.compute_rouge_scores(baseline_answer, system.current_reference)
            
            metrics_json["RAG ROUGE-L"] = round(rag_scores["rougeL"], 4)
            metrics_json["Baseline ROUGE-L"] = round(base_scores["rougeL"], 4)
            metrics_json["RAG ROUGE-1"] = round(rag_scores["rouge1"], 4)
            metrics_json["Baseline ROUGE-1"] = round(base_scores["rouge1"], 4)
        
        # Format Sources nicely
        source_html = "<div style='display: flex; flex-direction: column; gap: 10px;'>"
        for i, doc in enumerate(retrieved_docs):
            source_html += f"""
            <div style='background: #f8fafc; border-left: 4px solid #0ea5e9; padding: 12px; border-radius: 4px;'>
                <div style='font-weight: bold; color: #0284c7; margin-bottom: 4px;'>Source {i+1}: {doc['metadata'].get('title', 'Medical Research')}</div>
                <div style='font-size: 0.9em; color: #334155;'>{doc['text'][:400]}...</div>
            </div>
            """
        source_html += "</div>"
        
        return answer, baseline_answer, source_html, f"{latency:.2f}s", metrics_json
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return f"❌ Error: {str(e)}", "Error", "", "Error", None

def set_reference(question: str, reference: str):
    """Callback for examples to set the reference answer for the evaluator."""
    system.current_reference = reference
    return question

# --- UI Theme & CSS ---
# Using a clean, professional medical aesthetic
medical_theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

css = """
.medical-header { text-align: center; padding: 20px; background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 12px; margin-bottom: 20px; border: 1px solid #bae6fd; }
.stat-card { text-align: center; padding: 10px; background: white; border: 1px solid #e2e8f0; border-radius: 8px; }
.footer { text-align: center; color: #64748b; font-size: 0.8em; margin-top: 30px; }
"""

# --- Build App ---
with gr.Blocks(theme=medical_theme, css=css, title="MedBot Pro | Advanced Medical RAG") as demo:
    
    # 1. Header
    with gr.Row(elem_classes="medical-header"):
        with gr.Column():
            gr.Markdown("# 🏥 MedBot Pro: AI Medical Knowledge Assistant")
            gr.Markdown("### Advanced Retrieval-Augmented Generation (RAG) Platform for Biomedical Research")

    with gr.Row():
        # 2. Left Column: Control & Stats
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ System Control")
                corpus_slider = gr.Slider(minimum=50, maximum=1000, value=200, step=50, label="Knowledge Base Size", info="Number of Pubmed snippets to index.")
                init_btn = gr.Button("🚀 Initialize/Rebuild Index", variant="primary")
                status_box = gr.Textbox(label="Health Status", placeholder="System idle...", interactive=False)
            
            with gr.Group():
                gr.Markdown("### 📊 Performance Metrics")
                with gr.Row():
                    latency_box = gr.Label(label="Last Query Latency", value="0.00s")
            
            with gr.Accordion("ℹ️ How it works", open=False):
                gr.Markdown("""
                - **Retrieval**: Uses `all-MiniLM-L6-v2` Sentence-Transformers to create vector embeddings of medical literature stored in **ChromaDB**.
                - **LLM**: Powered by **Groq Llama-3** (8B/70B) for ultra-low latency contextual generation.
                - **Knowledge Base**: Curated from PubMed and MedQA subsets.
                """)

        # 3. Right Column: Interaction
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("💬 Medical Consultation"):
                    query_input = gr.Textbox(
                        label="Patient Case / Medical Question", 
                        placeholder="e.g., A 45-year-old male presents with acute substernal chest pain and diaphoresis...",
                        lines=3
                    )
                    submit_btn = gr.Button("Run Diagnostic RAG Pipeline", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🤖 MedBot Pro (RAG)")
                            answer_output = gr.Markdown("RAG response will appear here...")
                        with gr.Column():
                            gr.Markdown("#### 🌐 Baseline LLM (No RAG)")
                            baseline_output = gr.Markdown("Direct response will appear here...")
                    
                    with gr.Accordion("📚 Evidence & Source Documentation", open=True):
                        sources_output = gr.HTML("Initialize system to view sources...")

                with gr.TabItem("📈 Evaluation Benchmarks"):
                    gr.Markdown("Real-time metrics for the latest diagnostic run:")
                    metrics_display = gr.JSON(label="Query Benchmarks")
                    gr.Markdown("> **Note**: Full-scale benchmarking across the entire dataset is available via `python3 main.py`.")

    # 4. Examples & Footer
    examples_data = [
        ["What are the pathognomonic symptoms of a myocardial infarction?", "Chest pain, diaphoresis, nausea, and shortness of breath."],
        ["How does compression of the facial nerve at the stylomastoid foramen present?", "Facial asymmetry, difficulty closing eye, and lack of forehead wrinkles on affected side."],
        ["What is the first-line treatment for acute otitis media in pediatrics?", "Amoxicillin is typically the first-line antibiotic treatment."],
        ["Describe the mechanism of action for ACE inhibitors in hypertension.", "Inhibition of Angiotensin-Converting Enzyme, preventing conversion of Angiotensin I to Angiotensin II."]
    ]
    
    examples = gr.Examples(
        examples=examples_data,
        inputs=[query_input, gr.State("")], # dummy state to hold reference
        fn=set_reference,
        run_on_click=False,
        outputs=query_input
    )
    
    gr.Markdown("---")
    gr.Markdown("<div class='footer'>Developed for Production-Ready CV | Powered by ChromaDB, Groq & PyTorch</div>")

    # --- Event Handling ---
    init_btn.click(
        fn=initialize_system, 
        inputs=corpus_slider, 
        outputs=status_box
    )
    
    submit_btn.click(
        fn=process_query,
        inputs=query_input,
        outputs=[answer_output, baseline_output, sources_output, latency_box, metrics_display]
    )

if __name__ == "__main__":
    # Launch locally
    demo.launch(show_api=False)
