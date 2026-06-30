"""Run sample questions through the retriever and save the results.

Produces docs/examples/retrieval_examples.json as the Day 3
"saved retrieval examples" proof artifact.
"""

import json
from pathlib import Path

from retriever import Retriever
from sample_questions import SAMPLE_QUESTIONS

OUT_PATH = Path("docs/examples/retrieval_examples.json")
TOP_K = 5


def run() -> list[dict]:
    """Retrieve for every sample question, return a saveable list."""
    retriever = Retriever()
    examples = []

    for question in SAMPLE_QUESTIONS:
        hits = retriever.search(question, top_k=TOP_K)

        trimmed = []
        for h in hits:
            trimmed.append({
                "title": h.get("title", "?"),
                "source_url": h.get("source_url", "?"),
                "score": round(h["score"], 4),
                "preview": h["chunk_text"][:200].replace("\n", " "),
            })

        examples.append({"question": question, "results": trimmed})

    return examples


if __name__ == "__main__":
    examples = run()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(examples)} examples to {OUT_PATH}")