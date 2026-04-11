import time
from collections import defaultdict, deque
from typing import Callable

from flask import jsonify, request


class RequestGuard:
    def __init__(self, config):
        self.config = config
        self._request_window = defaultdict(deque)

    def _client_key(self) -> str:
        forwarded = request.headers.get("X-Forwarded-For", "")
        ip = forwarded.split(",")[0].strip() if forwarded else request.remote_addr or "unknown"
        return f"{ip}:{request.path}"

    def _is_api_path(self) -> bool:
        return request.path.startswith("/api/")

    def _is_exempt(self) -> bool:
        return request.path in {
            "/api/v1/health/live",
            "/api/v1/health/ready",
            "/health/live",
            "/health/ready",
        }

    def enforce_request_size(self):
        if request.content_length and request.content_length > self.config.max_request_bytes:
            return jsonify({"status": "error", "message": "Request payload too large"}), 413
        return None

    def enforce_json_contract(self):
        if self._is_api_path() and request.method in {"POST", "PUT", "PATCH"} and not request.is_json:
            return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 415
        return None

    def enforce_auth(self):
        if not self._is_api_path() or self._is_exempt():
            return None

        token = self.config.app_auth_token
        if not token:
            return None

        header = request.headers.get("Authorization", "")
        expected = f"Bearer {token}"
        if header != expected:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        return None

    def enforce_rate_limit(self):
        if not self._is_api_path() or self._is_exempt():
            return None
        now = time.time()
        key = self._client_key()
        bucket = self._request_window[key]

        while bucket and (now - bucket[0] > 60):
            bucket.popleft()
        if len(bucket) >= self.config.rate_limit_per_minute:
            return jsonify({"status": "error", "message": "Rate limit exceeded"}), 429

        bucket.append(now)
        return None

    def apply_security_headers(self, response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'self'; img-src 'self' data:; connect-src 'self'"

        origin = request.headers.get("Origin", "")
        allowed = self.config.cors_origins
        if origin and origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"

        return response

    def install(self, app):
        def _run_guards() -> tuple | None:
            for guard in (
                self.enforce_request_size,
                self.enforce_json_contract,
                self.enforce_auth,
                self.enforce_rate_limit,
            ):
                result = guard()
                if result is not None:
                    return result
            return None

        if hasattr(app, "before_request"):
            app.before_request(_run_guards)
        if hasattr(app, "after_request"):
            app.after_request(self.apply_security_headers)

        @app.route("/health/live", methods=["GET"])
        def legacy_live_health():
            return jsonify({"status": "ok"})

        @app.route("/health/ready", methods=["GET"])
        def legacy_ready_health():
            return jsonify({"status": "ok"})


def validate_payload(required_fields: list[str]) -> Callable:
    """Create a payload validator that checks required JSON field presence only.

    This validator only verifies whether required field names exist in the payload.
    It does not validate field data types, value ranges, or nested schema constraints.

    Args:
        required_fields: Field names that must be present in request payloads.

    Returns:
        Callable that accepts a payload dictionary and returns (is_valid, error_message).
    """

    def _validator(payload: dict) -> tuple[bool, str]:
        """Validate that all configured required fields exist in payload."""
        for field in required_fields:
            if field not in payload:
                return False, f"Missing field: {field}"
        return True, ""

    return _validator
