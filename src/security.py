"""Production-grade security middleware for MedRAG.

RequestGuard provides layered defense: request size limits, JSON contract validation,
Bearer token auth, IP-based rate limiting, and security headers.  Designed for
real Flask before_request / after_request hooks.
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict, deque
from typing import Callable, Optional, Tuple

from flask import jsonify, make_response, request


class RequestGuard:
    """Layered security guard for Flask applications."""

    def __init__(self, config):
        self.config = config
        self._request_window: defaultdict[str, deque[float]] = defaultdict(deque)
        self._banned_ips: set[str] = set()

    # ------------------------------------------------------------------
    # Client identity
    # ------------------------------------------------------------------
    def _client_key(self) -> str:
        forwarded = request.headers.get("X-Forwarded-For", "")
        ip = forwarded.split(",")[0].strip() if forwarded else request.remote_addr or "unknown"
        return f"{ip}:{request.path}"

    def _client_ip(self) -> str:
        forwarded = request.headers.get("X-Forwarded-For", "")
        return forwarded.split(",")[0].strip() if forwarded else request.remote_addr or "unknown"

    # ------------------------------------------------------------------
    # Path classification
    # ------------------------------------------------------------------
    def _is_api_path(self) -> bool:
        return request.path.startswith("/api/")

    def _is_exempt(self) -> bool:
        return request.path in {
            "/api/v1/health/live",
            "/api/v1/health/ready",
            "/health/live",
            "/health/ready",
            "/",
        }

    # ------------------------------------------------------------------
    # Individual guards (return None = pass, return Response = block)
    # ------------------------------------------------------------------
    def enforce_request_size(self) -> Optional[Tuple]:
        if request.content_length and request.content_length > self.config.max_request_bytes:
            return jsonify({"status": "error", "message": "Request payload too large"}), 413
        return None

    def enforce_json_contract(self) -> Optional[Tuple]:
        if (
            self._is_api_path()
            and request.method in {"POST", "PUT", "PATCH"}
            and not request.is_json
        ):
            return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 415
        return None

    def enforce_auth(self) -> Optional[Tuple]:
        if not self._is_api_path() or self._is_exempt():
            return None

        token = getattr(self.config, "app_auth_token", "") or ""
        if not token:
            return None  # Auth not configured

        header = request.headers.get("Authorization", "")
        expected = f"Bearer {token}"
        if not header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Missing or malformed Authorization header"}), 401
        if header != expected:
            return jsonify({"status": "error", "message": "Invalid token"}), 401
        return None

    def enforce_rate_limit(self) -> Optional[Tuple]:
        if not self._is_api_path() or self._is_exempt():
            return None

        ip = self._client_ip()
        if ip in self._banned_ips:
            return jsonify({"status": "error", "message": "Access denied"}), 403

        now = time.time()
        key = self._client_key()
        bucket = self._request_window[key]

        # Evict stale entries
        window_seconds = 60
        while bucket and (now - bucket[0] > window_seconds):
            bucket.popleft()

        limit = getattr(self.config, "rate_limit_per_minute", 60)
        if len(bucket) >= limit:
            return jsonify({"status": "error", "message": "Rate limit exceeded"}), 429

        bucket.append(now)
        return None

    # ------------------------------------------------------------------
    # Response hardening
    # ------------------------------------------------------------------
    def apply_security_headers(self, response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), payment=(), usb=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; "
            "script-src 'self'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["X-Content-Security-Policy"] = response.headers["Content-Security-Policy"]
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # CORS
        origin = request.headers.get("Origin", "")
        allowed = getattr(self.config, "cors_origins", [])
        if origin and origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-Requested-With"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Max-Age"] = "86400"

        return response

    # ------------------------------------------------------------------
    # Flask integration
    # ------------------------------------------------------------------
    def install(self, app):
        """Register guards with a Flask application."""

        @app.before_request
        def _run_guards():
            for guard in (
                self.enforce_request_size,
                self.enforce_json_contract,
                self.enforce_auth,
                self.enforce_rate_limit,
            ):
                result = guard()
                if result is not None:
                    # Flask before_request expects Response or None
                    payload, status = result
                    response = make_response(payload, status)
                    return response
            return None

        @app.after_request
        def _harden_response(response):
            return self.apply_security_headers(response)

        # Legacy health routes (also covered by _is_exempt)
        @app.route("/health/live", methods=["GET"])
        def legacy_live_health():
            return jsonify({"status": "ok"})

        @app.route("/health/ready", methods=["GET"])
        def legacy_ready_health():
            ready = getattr(app, "config", {}).get("med_service", None)
            if ready and hasattr(ready, "health_ready"):
                is_ready = ready.health_ready()
                return jsonify({"status": "ok" if is_ready else "initializing"}), (200 if is_ready else 503)
            return jsonify({"status": "ok"})


# ------------------------------------------------------------------
# Payload validation
# ------------------------------------------------------------------

def validate_payload(required_fields: list[str]) -> Callable:
    """Create a payload validator that checks required JSON field presence."""

    def _validator(payload: dict) -> tuple[bool, str]:
        if not isinstance(payload, dict):
            return False, "Payload must be a JSON object"
        for field in required_fields:
            if field not in payload:
                return False, f"Missing field: {field}"
        return True, ""

    return _validator
