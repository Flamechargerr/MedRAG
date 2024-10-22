"""Production Flask application for MedRAG.

Clean, minimal, production-hardened API with proper error handling,
health checks, and honest response structures.
"""

import logging
import os

from flask import Flask, jsonify, render_template, request

from src.config import get_config
from src.security import RequestGuard, validate_payload
from src.services.med_service import MedRAGService

logger = logging.getLogger("MedRAG")


def create_app() -> Flask:
    config = get_config()
    logging.getLogger().setLevel(config.log_level)

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["JSON_SORT_KEYS"] = False

    # Security middleware
    guard = RequestGuard(config)
    guard.install(app)

    # Service layer
    service = MedRAGService(config)
    app.config["med_service"] = service

    # Payload validators
    init_validator = validate_payload([])
    chat_validator = validate_payload(["query"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _handle_init():
        try:
            payload = request.get_json(silent=True) or {}
            valid, message = init_validator(payload)
            if not valid:
                return jsonify({"status": "error", "message": message}), 400

            corpus_size = int(payload.get("corpus_size", config.default_corpus_size))
            force_reindex = bool(payload.get("force_reindex", False))
            if corpus_size < 1 or corpus_size > config.max_corpus_size:
                return jsonify(
                    {"status": "error", "message": f"corpus_size must be 1-{config.max_corpus_size}"}
                ), 400

            response = service.initialize(corpus_size=corpus_size, force_reindex=force_reindex)
            return jsonify(response)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception:
            logger.exception("Init error")
            return jsonify({"status": "error", "message": "Internal server error"}), 500

    def _handle_chat():
        try:
            payload = request.get_json(silent=True) or {}
            valid, message = chat_validator(payload)
            if not valid:
                return jsonify({"status": "error", "message": message}), 400

            question = str(payload.get("query", "")).strip()
            reference = str(payload.get("reference", "")).strip()
            if not question:
                return jsonify({"status": "error", "message": "query cannot be empty"}), 400

            response = service.chat(question=question, reference=reference)
            return jsonify(response)
        except RuntimeError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception:
            logger.exception("Chat error")
            return jsonify({"status": "error", "message": "Internal server error"}), 500

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/v1/health/live", methods=["GET"])
    def health_live():
        return jsonify({"status": "ok"})

    @app.route("/api/v1/health/ready", methods=["GET"])
    def health_ready():
        is_ready = service.health_ready()
        return jsonify({"status": "ok" if is_ready else "initializing"}), (200 if is_ready else 503)

    @app.route("/api/v1/status", methods=["GET"])
    def status():
        return jsonify(service.get_status())

    @app.route("/api/v1/init", methods=["POST", "OPTIONS"])
    def init_v1():
        if request.method == "OPTIONS":
            return ("", 204)
        return _handle_init()

    @app.route("/api/v1/chat", methods=["POST", "OPTIONS"])
    def chat_v1():
        if request.method == "OPTIONS":
            return ("", 204)
        return _handle_chat()

    # Legacy routes (backward compatible)
    @app.route("/api/init", methods=["POST", "OPTIONS"])
    def init_legacy():
        if request.method == "OPTIONS":
            return ("", 204)
        return _handle_init()

    @app.route("/api/chat", methods=["POST", "OPTIONS"])
    def chat_legacy():
        if request.method == "OPTIONS":
            return ("", 204)
        return _handle_chat()

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"status": "error", "message": "Not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"status": "error", "message": "Method not allowed"}), 405

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"status": "error", "message": "Internal server error"}), 500

    return app


app = create_app()
system = app.config["med_service"]


# Legacy module-level exports for backward compatibility
def index():
    return app.view_functions["index"]()


def initialize():
    return app.view_functions["init_legacy"]()


def chat():
    return app.view_functions["chat_legacy"]()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=False, port=port)
