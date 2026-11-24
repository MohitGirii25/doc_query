"""REST API with validation, least-privilege auth, human review, telemetry and a dashboard.

Public:    GET /  (dashboard)   GET /health   GET /metrics
           POST /ask {"question": "..."}      GET /answers/<id>
Reviewer:  GET /reviews   POST /reviews/<id>/decision {"approved": true|false}
           (requires header X-API-Key matching the reviewer key)
"""
import hmac
import logging
import time
from pathlib import Path

from flask import Flask, g, jsonify, request, send_file

from .chatbot import DocuQuery
from .review import PENDING, ReviewQueue

logger = logging.getLogger("docuquery")
MAX_QUESTION_LENGTH = 500
STATIC_DIR = Path(__file__).parent / "static"


def create_app(data_dir="data", reviewer_key=None, review_threshold=0.2):
    app = Flask(__name__)
    bot = DocuQuery(data_dir)
    queue = ReviewQueue()
    stats = {"requests": 0, "errors": 0, "total_ms": 0.0, "pending_reviews": 0}

    def reviewer_error():
        """Least privilege: only holders of the reviewer key may use review endpoints."""
        if not reviewer_key:
            return jsonify(error="reviewer access is not configured"), 503
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied, reviewer_key):
            return jsonify(error="unauthorized"), 401
        return None

    @app.before_request
    def start_timer():
        g.start = time.perf_counter()

    @app.after_request
    def record_telemetry(response):
        latency_ms = (time.perf_counter() - g.start) * 1000
        stats["requests"] += 1
        stats["total_ms"] += latency_ms
        if response.status_code >= 400:
            stats["errors"] += 1
        logger.info(
            "method=%s path=%s status=%s latency_ms=%.1f",
            request.method, request.path, response.status_code, latency_ms,
        )
        return response

    @app.errorhandler(Exception)
    def graceful_failure(exc):
        logger.exception("unhandled error")
        return jsonify(error="Something went wrong. Please try again later."), 500

    @app.get("/")
    def dashboard():
        return send_file(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/metrics")
    def metrics():
        n = stats["requests"]
        avg = round(stats["total_ms"] / n, 2) if n else 0.0
        return jsonify(
            requests=n, errors=stats["errors"], avg_latency_ms=avg,
            pending_reviews=len(queue.pending()),
        )

    @app.post("/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            return jsonify(error="'question' is required"), 400
        question = question.strip()
        if len(question) > MAX_QUESTION_LENGTH:
            return jsonify(error=f"question must be {MAX_QUESTION_LENGTH} characters or fewer"), 400
        result = bot.ask(question)
        if result["confidence"] < review_threshold and result["sources"]:
            review_id = queue.submit(result)
            return jsonify(
                status="pending_review", review_id=review_id,
                message="Low-confidence answer sent for human review.",
            ), 202
        return jsonify(status="answered", **result)

    @app.get("/answers/<int:review_id>")
    def answer_status(review_id):
        item = queue.get(review_id)
        if item is None:
            return jsonify(error="not found"), 404
        if item["status"] == PENDING:
            return jsonify(status=PENDING), 202
        if item["status"] == "approved":
            return jsonify(status="approved", answer=item["answer"], sources=item["sources"])
        return jsonify(status="rejected", message="A reviewer rejected this answer.")

    @app.get("/reviews")
    def list_reviews():
        denied = reviewer_error()
        if denied:
            return denied
        return jsonify(pending=queue.pending())

    @app.post("/reviews/<int:review_id>/decision")
    def decide(review_id):
        denied = reviewer_error()
        if denied:
            return denied
        approved = (request.get_json(silent=True) or {}).get("approved")
        if not isinstance(approved, bool):
            return jsonify(error="'approved' must be true or false"), 400
        try:
            item = queue.decide(review_id, approved)
        except KeyError:
            return jsonify(error="not found"), 404
        except ValueError:
            return jsonify(error="already decided"), 409
        return jsonify(id=item["id"], status=item["status"])

    return app
