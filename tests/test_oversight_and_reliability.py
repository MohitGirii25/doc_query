import requests

from docuquery import llm
from docuquery.api import create_app
from docuquery.chatbot import DocuQuery
from docuquery.evaluate import evaluate

KEY = "test-reviewer-key"


def make_client(**kwargs):
    return create_app("data", **kwargs).test_client()


# ---- input validation and failure cases ----
def test_rejects_empty_non_string_and_too_long_questions():
    client = make_client()
    assert client.post("/ask", json={"question": "   "}).status_code == 400
    assert client.post("/ask", json={"question": 123}).status_code == 400
    assert client.post("/ask", json={"question": "a" * 501}).status_code == 400


def test_unhandled_error_returns_graceful_500(monkeypatch):
    app = create_app("data")
    monkeypatch.setattr(DocuQuery, "ask", lambda self, q, k=3: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = app.test_client().post("/ask", json={"question": "hello"})
    assert resp.status_code == 500
    assert "went wrong" in resp.get_json()["error"]


def test_llm_failure_falls_back_to_best_match(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    def boom(*args, **kwargs):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(llm.requests, "post", boom)
    result = DocuQuery("data").ask("How many days of sick leave do employees get?")
    assert "LLM call failed" in result["answer"] and "12 days of sick leave" in result["answer"]


# ---- human oversight (low-confidence answers need approval) ----
def test_low_confidence_answer_waits_for_human_review_then_is_released():
    client = make_client(reviewer_key=KEY, review_threshold=0.99)  # force review
    resp = client.post("/ask", json={"question": "How many days of sick leave do employees get?"})
    assert resp.status_code == 202 and resp.get_json()["status"] == "pending_review"
    review_id = resp.get_json()["review_id"]
    assert "answer" not in resp.get_json()                       # answer is withheld
    assert client.get(f"/answers/{review_id}").status_code == 202

    headers = {"X-API-Key": KEY}
    pending = client.get("/reviews", headers=headers).get_json()["pending"]
    assert pending[0]["id"] == review_id and "12 days" in pending[0]["answer"]

    decision = client.post(f"/reviews/{review_id}/decision", json={"approved": True}, headers=headers)
    assert decision.get_json()["status"] == "approved"
    released = client.get(f"/answers/{review_id}").get_json()
    assert released["status"] == "approved" and "12 days" in released["answer"]


def test_rejected_answer_is_never_released():
    client = make_client(reviewer_key=KEY, review_threshold=0.99)
    review_id = client.post("/ask", json={"question": "What are the core hours?"}).get_json()["review_id"]
    client.post(f"/reviews/{review_id}/decision", json={"approved": False}, headers={"X-API-Key": KEY})
    body = client.get(f"/answers/{review_id}").get_json()
    assert body["status"] == "rejected" and "answer" not in body


def test_decision_cannot_be_changed_twice():
    client = make_client(reviewer_key=KEY, review_threshold=0.99)
    rid = client.post("/ask", json={"question": "What are the core hours?"}).get_json()["review_id"]
    url, headers = f"/reviews/{rid}/decision", {"X-API-Key": KEY}
    assert client.post(url, json={"approved": True}, headers=headers).status_code == 200
    assert client.post(url, json={"approved": False}, headers=headers).status_code == 409


# ---- authentication and least privilege ----
def test_review_endpoints_require_the_reviewer_key():
    client = make_client(reviewer_key=KEY)
    assert client.get("/reviews").status_code == 401
    assert client.get("/reviews", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/reviews/1/decision", json={"approved": True}).status_code == 401
    assert client.get("/reviews", headers={"X-API-Key": KEY}).status_code == 200


def test_review_endpoints_are_disabled_without_a_configured_key():
    assert make_client().get("/reviews", headers={"X-API-Key": "anything"}).status_code == 503


def test_decision_validates_payload_and_unknown_ids():
    client = make_client(reviewer_key=KEY)
    headers = {"X-API-Key": KEY}
    assert client.post("/reviews/1/decision", json={"approved": "yes"}, headers=headers).status_code == 400
    assert client.post("/reviews/999/decision", json={"approved": True}, headers=headers).status_code == 404
    assert client.get("/answers/999").status_code == 404


def test_confident_answers_are_returned_directly():
    resp = make_client().post("/ask", json={"question": "How many days of sick leave do employees get?"})
    assert resp.status_code == 200 and resp.get_json()["status"] == "answered"


# ---- observability and dashboard accessibility (WCAG basics) ----
def test_metrics_include_pending_reviews():
    client = make_client(reviewer_key=KEY, review_threshold=0.99)
    client.post("/ask", json={"question": "What are the core hours?"})
    assert client.get("/metrics").get_json()["pending_reviews"] == 1


def test_dashboard_uses_accessible_markup():
    html = make_client().get("/").get_data(as_text=True)
    for marker in ['<html lang="en">', "<main", 'for="question"', 'aria-live="polite"',
                   "Skip to main content", "<caption>", 'scope="col"', ":focus-visible"]:
        assert marker in html, marker


# ---- evaluation of retrieval reliability ----
def test_retrieval_hit_rate_meets_threshold():
    result = evaluate(DocuQuery("data"), "eval/eval_questions.csv", k=3)
    assert result["total"] >= 10 and result["hit_rate"] >= 0.8
