import logging

from docuquery.api import create_app
from docuquery.chatbot import DocuQuery
from docuquery.chunker import chunk_text
from docuquery.loader import load_csv_records, load_text_documents
from docuquery.retriever import Retriever


def test_chunk_text_overlaps_and_covers_all_words():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, size=40, overlap=10)
    assert len(chunks) >= 3
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert chunks[-1].endswith("w99")


def test_loaders_read_unstructured_and_structured_data():
    assert len(load_text_documents("data/docs")) == 3
    rows = load_csv_records("data/faq.csv")
    assert len(rows) == 4 and "probation" in rows[0]["text"].lower()


def test_retriever_ranks_relevant_source_first():
    chunks = [
        {"source": "a", "text": "vacation days and sick leave policy"},
        {"source": "b", "text": "code review pull request approval"},
    ]
    top = Retriever(chunks).search("how many sick leave days", k=1)[0]
    assert top["source"] == "a"


def test_chatbot_answers_from_documents():
    result = DocuQuery("data").ask("How many days of sick leave do employees get?")
    assert "12 days of sick leave" in result["answer"]
    assert "leave_policy.txt" in result["sources"]


def test_chatbot_handles_structured_csv_question():
    result = DocuQuery("data").ask("How do I reset my password?")
    assert "faq.csv#row3" in result["sources"]


def test_unknown_question_returns_dont_know():
    result = DocuQuery("data").ask("zzzz qqqq xxxx")
    assert "don't know" in result["answer"].lower()


def test_api_ask_and_validation():
    client = create_app("data").test_client()
    assert client.get("/health").get_json() == {"status": "ok"}
    ok = client.post("/ask", json={"question": "What is the hybrid work model?"})
    assert ok.status_code == 200 and "three days" in ok.get_json()["answer"]
    assert client.post("/ask", json={}).status_code == 400


def test_metrics_endpoint_tracks_requests_and_errors():
    client = create_app("data").test_client()
    client.post("/ask", json={"question": "What is the hybrid work model?"})
    client.post("/ask", json={})  # invalid request -> counted as an error
    metrics = client.get("/metrics").get_json()
    assert metrics["requests"] >= 2
    assert metrics["errors"] >= 1
    assert metrics["avg_latency_ms"] >= 0


def test_requests_are_logged_with_latency(caplog):
    client = create_app("data").test_client()
    with caplog.at_level(logging.INFO, logger="docuquery"):
        client.get("/health")
    assert "path=/health" in caplog.text and "latency_ms=" in caplog.text
