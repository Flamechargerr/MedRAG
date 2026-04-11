import logging

from flask import Flask, jsonify, render_template, request

from src.config import Config, get_config
from src.security import RequestGuard, validate_payload
from src.services.med_service import MedRAGService

logger = logging.getLogger("MedBot-Flask")


def create_app() -> Flask:
    config = get_config()
    logging.getLogger().setLevel(config.log_level)

    app = Flask(__name__, static_folder="static", template_folder="templates")
    guard = RequestGuard(config)
    guard.install(app)

    service = MedRAGService(config)
    app.config["med_service"] = service

    init_payload_validator = validate_payload([])
    chat_payload_validator = validate_payload(["query"])

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/v1/health/live", methods=["GET"])
    def live_health():
        return jsonify({"status": "ok"})

    @app.route("/api/v1/health/ready", methods=["GET"])
    def ready_health():
        return jsonify({"status": "ok" if service.health_ready() else "initializing"}), (200 if service.health_ready() else 503)

    def _initialize_impl():
        try:
            payload = request.json or {}
            valid, message = init_payload_validator(payload)
            if not valid:
                return jsonify({"status": "error", "message": message}), 400

            corpus_size = int(payload.get("corpus_size", config.default_corpus_size))
            force_reindex = bool(payload.get("force_reindex", False))
            response = service.initialize(corpus_size=corpus_size, force_reindex=force_reindex)
            return jsonify(response)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:
            logger.exception("Init error")
            return jsonify({"status": "error", "message": "Internal server error"}), 500

    def _chat_impl():
        try:
            payload = request.json or {}
            valid, message = chat_payload_validator(payload)
            if not valid:
                return jsonify({"status": "error", "message": message}), 400

            question = str(payload.get("query", ""))
            reference = str(payload.get("reference", ""))
            response = service.chat(question=question, reference=reference)
            return jsonify(response)
        except RuntimeError:
            return jsonify({"status": "error", "message": "Service is not initialized"}), 400
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:
            logger.exception("Chat error")
            return jsonify({"status": "error", "message": "Internal server error"}), 500

    @app.route("/api/v1/init", methods=["POST", "OPTIONS"])
    def initialize_v1():
        if request.method == "OPTIONS":
            return ("", 204)
        return _initialize_impl()

    @app.route("/api/v1/chat", methods=["POST", "OPTIONS"])
    def chat_v1():
        if request.method == "OPTIONS":
            return ("", 204)
        return _chat_impl()

    # Backward-compatible routes
    @app.route("/api/init", methods=["POST", "OPTIONS"])
    def initialize_legacy():
        if request.method == "OPTIONS":
            return ("", 204)
        return _initialize_impl()

    @app.route("/api/chat", methods=["POST", "OPTIONS"])
    def chat_legacy():
        if request.method == "OPTIONS":
            return ("", 204)
        return _chat_impl()

    return app


app = create_app()
system = app.config["med_service"]


def index():
    return app.view_functions["index"]()


def initialize():
    return app.view_functions["initialize_legacy"]()


def chat():
    return app.view_functions["chat_legacy"]()


if __name__ == "__main__":
    Config.load_env()
    app.run(host="0.0.0.0", debug=False, port=5000)
