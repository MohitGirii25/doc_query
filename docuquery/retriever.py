
"""TF-IDF retriever with improved relevance filtering."""

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("No chunks to index")

        self.chunks = chunks

        # Keep standard English stop words and add domain-specific words.
        custom_stop_words = list(
            ENGLISH_STOP_WORDS.union({
                "company",
                "employee",
                "employees"
            })
        )

        self.vectorizer = TfidfVectorizer(
            stop_words=custom_stop_words,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )

    def search(self, query, k=3, min_score=0.10):
        query_vec = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vec, self.matrix
        ).ravel()

        ranked = scores.argsort()[::-1]

        results = []

        for index in ranked:
            score = float(scores[index])

            if score < min_score:
                break

            results.append({
                **self.chunks[index],
                "score": round(score, 4)
            })

            if len(results) >= k:
                break

        return results
