
"""Run the chatbot in the terminal: python main.py (or API: python main.py --api)"""

import logging
import os
import sys

from docuquery.api import create_app
from docuquery.chatbot import DocuQuery


def run_cli():
    bot = DocuQuery("data")
    print("DocuQuery ready. Ask a question (type 'exit' to quit).")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        result = bot.ask(question)

        print(f"Bot: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")
        print(f"Confidence: {result['confidence']}")


if __name__ == "__main__":
    if "--api" in sys.argv:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s"
        )
        create_app(
            "data",
            reviewer_key=os.getenv("REVIEWER_API_KEY")
        ).run(port=5000)
    else:
        run_cli()
