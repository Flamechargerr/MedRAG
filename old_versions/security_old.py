"""Basic security middleware."""
from flask import jsonify, request
from collections import defaultdict, deque
import time

class RequestGuard:
    def __init__(self, config):
        self.config = config
        self._window = defaultdict(deque)
    
    def enforce_rate_limit(self):
        key = f"{request.remote_addr}:{request.path}"
        bucket = self._window[key]
        now = time.time()
        while bucket and now - bucket[0] > 60: bucket.popleft()
        if len(bucket) >= getattr(self.config, 'rate_limit_per_minute', 60):
            return jsonify({"error": "Rate limited"}), 429
        bucket.append(now)
        return None
