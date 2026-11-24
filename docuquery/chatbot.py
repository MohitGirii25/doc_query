
"""DocuQuery: connects loading, chunking, retrieval and answer generation."""

from pathlib import Path

from .chunker import chunk_records
from .llm import generate_answer
from .loader import load_csv_records, load_text_documents
from .retriever import Retriever


class DocuQuery:

    def __init__(self, data_dir="data"):
        data_dir = Path(data_dir)

        records = load_text_documents(data_dir / "docs")

        for csv_file in sorted(data_dir.glob("*.csv")):
            records += load_csv_records(csv_file)

        self.retriever = Retriever(
    chunk_records(records, size=120, overlap=20)
)

    def ask(self, question, k=3):
        contexts = self.retriever.search(question, k=k)

        if not contexts:
            return {
                "question": question,
                "answer": "I don't know. I couldn't find relevant information in the uploaded documents.",
                "sources": [],
                "confidence": 0.0,
            }

        return {
            "question": question,
            "answer": generate_answer(question, contexts),
            "sources": sorted({c["source"] for c in contexts}),
            "confidence": contexts[0]["score"],
        }
