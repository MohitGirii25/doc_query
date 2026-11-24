# DocuQuery – RAG Document Q&A Chatbot (Python)

A small Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own
documents. It handles **unstructured data** (.txt / .md files) and **structured data** (CSV).

## How it works
1. **Load** text files and CSV rows into one record format (`loader.py`, pandas).
2. **Chunk** long text into small overlapping pieces (`chunker.py`).
3. **Retrieve** the most relevant chunks with TF-IDF + cosine similarity (`retriever.py`, scikit-learn).
4. **Generate** an answer from the retrieved context with prompt engineering (`llm.py`).
   Uses Gemini or OpenAI over REST if configured; otherwise an offline extractive fallback.
5. **Serve** through a terminal chatbot (`main.py`) or a Flask REST API (`api.py`).

## Safety, oversight and reliability
- **Human-in-the-loop:** answers below a confidence threshold are withheld and queued for review; an
  authorized reviewer approves or rejects them (`/reviews`, `/reviews/<id>/decision`).
- **Least privilege:** review endpoints need the `X-API-Key` header (set `REVIEWER_API_KEY`); without it they are disabled.
- **Input validation and graceful failure:** empty, non-text or over-500-character questions are rejected;
  LLM/network failures fall back to the best retrieved passage; unexpected errors return a safe JSON message.
- **Evaluation:** `python -m docuquery.evaluate` measures retrieval hit rate@3 on a labeled question set (`eval/`).
- **Accessible dashboard:** `GET /` serves a page with semantic landmarks, labels, ARIA live regions, a skip link,
  visible focus and keyboard support (WCAG basics).

## Testing, telemetry and CI
- **Automated tests:** pytest unit, functional and API tests with coverage (`pytest --cov=docuquery`).
- **Telemetry:** every request is logged with method, path, status and latency; `GET /metrics`
  reports request count, error count and average latency.
- **CI:** a GitHub Actions workflow (`.github/workflows/ci.yml`) runs the test suite on every push.

## Run
```bash
pip install -r requirements.txt
python main.py            # terminal chatbot
python main.py --api      # REST API on http://localhost:5000
pytest --cov=docuquery    # run the automated tests with coverage
python -m docuquery.evaluate   # retrieval hit-rate evaluation
REVIEWER_API_KEY=secret python main.py --api   # enable the reviewer endpoints
```

Optional LLM answers:
```bash
export LLM_PROVIDER=gemini   GEMINI_API_KEY=your_key     # or: openai / OPENAI_API_KEY
```

API example:
```bash
curl -X POST localhost:5000/ask -H "Content-Type: application/json" \
     -d '{"question": "How many sick leave days do employees get?"}'
```

## Stack
Python, pandas, scikit-learn (TF-IDF), Flask, requests, pytest, Git.
