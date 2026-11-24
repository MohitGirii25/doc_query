"""Human-in-the-loop review queue.

Answers the bot is not confident about are NOT returned to the user directly. They wait in
this queue until an authorized reviewer approves or rejects them.
"""
import itertools
import threading

PENDING, APPROVED, REJECTED = "pending", "approved", "rejected"


class ReviewQueue:
    def __init__(self):
        self._items = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def submit(self, result):
        with self._lock:
            review_id = next(self._ids)
            self._items[review_id] = {"id": review_id, "status": PENDING, **result}
            return review_id

    def get(self, review_id):
        return self._items.get(review_id)

    def pending(self):
        return [item for item in self._items.values() if item["status"] == PENDING]

    def decide(self, review_id, approved):
        with self._lock:
            item = self._items.get(review_id)
            if item is None:
                raise KeyError(review_id)
            if item["status"] != PENDING:
                raise ValueError("already decided")
            item["status"] = APPROVED if approved else REJECTED
            return item
