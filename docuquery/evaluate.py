"""Measure retrieval reliability on a small labeled question set (hit rate @ k)."""
import csv

from .chatbot import DocuQuery


def evaluate(bot, eval_path="eval/eval_questions.csv", k=3):
    total = hits = 0
    misses = []
    with open(eval_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            sources = {c["source"] for c in bot.retriever.search(row["question"], k=k)}
            if row["expected_source"] in sources:
                hits += 1
            else:
                misses.append(row["question"])
    return {"total": total, "hits": hits, "hit_rate": round(hits / total, 3) if total else 0.0, "misses": misses}


if __name__ == "__main__":
    result = evaluate(DocuQuery("data"))
    print(f"Retrieval hit rate @3: {result['hits']}/{result['total']} = {result['hit_rate']:.0%}")
    for q in result["misses"]:
        print("  missed:", q)
