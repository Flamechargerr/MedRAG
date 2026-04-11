import importlib
import sys
import types
import unittest


def _install_app_dependency_stubs():
    flask_module = types.ModuleType("flask")

    class Flask:
        def __init__(self, *args, **kwargs):
            self.routes = {}
            self.view_functions = {}
            self.config = {}
            self.before_request_handlers = []
            self.after_request_handlers = []

        def route(self, path, methods=None):
            def decorator(func):
                self.routes[(path, tuple(methods or ["GET"]))] = func
                self.view_functions[func.__name__] = func
                return func

            return decorator

        def before_request(self, func):
            self.before_request_handlers.append(func)
            return func

        def after_request(self, func):
            self.after_request_handlers.append(func)
            return func

    flask_module.Flask = Flask
    flask_module.request = types.SimpleNamespace(
        json=None,
        method="POST",
        headers={},
        content_length=0,
        is_json=True,
        path="/api/init",
        remote_addr="127.0.0.1",
    )
    flask_module.jsonify = lambda payload: payload
    flask_module.render_template = lambda template_name: f"rendered:{template_name}"
    sys.modules["flask"] = flask_module

    config_module = types.ModuleType("src.config")

    class Config:
        DEVICE = "cpu"

        @classmethod
        def load_env(cls):
            cls.DEVICE = "cpu"

    class RuntimeConfig:
        environment = "local"
        log_level = "INFO"
        api_key = ""
        huggingface_api_key = ""
        app_auth_token = ""
        cors_origins = ["http://127.0.0.1:5000"]
        rate_limit_per_minute = 200
        max_request_bytes = 65536
        max_query_chars = 2000
        default_corpus_size = 200
        max_corpus_size = 2000
        retrieval_top_k = 3
        faiss_db_dir = "./faiss_db"
        runtime_state_path = "/tmp/test-runtime-state.json"
        llm_max_retries = 0

    config_module.OPENCALL_LLM_KEY = ""
    config_module.Config = Config
    config_module.get_config = lambda: RuntimeConfig()
    sys.modules["src.config"] = config_module

    data_loader_module = types.ModuleType("src.data_loader")
    data_loader_module.load_medqa_data = lambda num_eval_questions=5: ([], [])

    def load_medical_corpus(fallback_dataset=None, max_docs=200, **kwargs):
        return [{"text": "Document evidence for testing", "id": "1", "title": "Test Source"}]

    data_loader_module.load_medical_corpus = load_medical_corpus
    sys.modules["src.data_loader"] = data_loader_module

    vector_store_module = types.ModuleType("src.retrieval.langchain_faiss_store")

    class _Index:
        ntotal = 0

    class _VectorStore:
        index = _Index()

    class LangchainFAISSStore:
        def __init__(self, db_dir="./faiss_db", device="cpu"):
            self.docs = []
            self.vector_store = None

        def add_documents(self, documents):
            self.docs.extend(documents)
            self.vector_store = _VectorStore()
            self.vector_store.index.ntotal = len(self.docs)

        def retrieve(self, query, top_k=3):
            return [{"text": "retrieved evidence", "metadata": {"title": "Stub Source"}}]

    vector_store_module.LangchainFAISSStore = LangchainFAISSStore
    sys.modules["src.retrieval.langchain_faiss_store"] = vector_store_module

    llm_module = types.ModuleType("src.generation.llm_generators")

    class GroqGenerator:
        def __init__(self, api_key, max_retries=0):
            self.api_key = api_key

        def generate(self, question, retrieved_docs):
            return "RAG answer based on retrieved evidence"

        def generate_no_context(self, question):
            return "Baseline answer without retrieval"

    llm_module.GroqGenerator = GroqGenerator
    sys.modules["src.generation.llm_generators"] = llm_module

    metrics_module = types.ModuleType("src.evaluation.metrics")

    class RAGEvaluator:
        def compute_rouge_scores(self, generated_answer, reference_answer):
            if "RAG" in str(generated_answer):
                return {"rouge1": 0.8, "rouge2": 0.7, "rougeL": 0.75}
            return {"rouge1": 0.5, "rouge2": 0.4, "rougeL": 0.45}

    metrics_module.RAGEvaluator = RAGEvaluator
    sys.modules["src.evaluation.metrics"] = metrics_module


class FlaskApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_app_dependency_stubs()
        if "app" in sys.modules:
            cls.app_module = importlib.reload(sys.modules["app"])
        else:
            cls.app_module = importlib.import_module("app")

    def setUp(self):
        self.app_module.system.is_initialized = False
        self.app_module.system.vector_store = None
        self.app_module.system.llm_gen = None
        self.app_module.request.json = None
        self.app_module.request.method = "POST"

    def test_index_route(self):
        response = self.app_module.index()
        self.assertEqual(response, "rendered:index.html")

    def test_chat_requires_initialization(self):
        self.app_module.request.json = {"query": "What is flu?"}
        payload, status = self.app_module.chat()
        self.assertEqual(status, 400)
        self.assertEqual(payload["status"], "error")

    def test_init_route_sets_initialized_state(self):
        self.app_module.request.json = {"corpus_size": 3}
        payload = self.app_module.initialize()
        self.assertEqual(payload["status"], "success")
        self.assertTrue(self.app_module.system.is_initialized)

    def test_chat_returns_metrics_sources_and_answers(self):
        self.app_module.request.json = {"corpus_size": 2}
        self.app_module.initialize()
        self.app_module.request.json = {
            "query": "How should dehydration be treated?",
            "reference": "oral rehydration",
        }
        payload = self.app_module.chat()
        if isinstance(payload, tuple):
            payload, status = payload
            self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "success")
        self.assertIn("answer", payload)
        self.assertIn("baseline_answer", payload)
        self.assertTrue(payload["sources"])
        self.assertIn("RAG_ROUGE_L", payload["metrics"])
        self.assertIn("Baseline_ROUGE_L", payload["metrics"])
        self.assertIn("Accuracy_Improvement", payload["metrics"])

    def test_chat_requires_query(self):
        self.app_module.request.json = {"corpus_size": 2}
        self.app_module.initialize()
        self.app_module.request.json = {"query": ""}
        payload, status = self.app_module.chat()
        self.assertEqual(status, 400)
        self.assertIn("message", payload)

    def test_v1_health_ready_status(self):
        response, status = self.app_module.app.view_functions["ready_health"]()
        self.assertIn(status, {200, 503})
        self.assertIn("status", response)


if __name__ == "__main__":
    unittest.main()
