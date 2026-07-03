import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "Retrieval")
)
from retriever import Retriever
from sample_questions import SAMPLE_QUESTIONS

TOP_K = 15
OUT_PATH = Path("data/eval/golden_relevance.jsonl")


def format_candidate(rank: int, chunk: dict) -> str:
    """Render one retrieved chunk for manual review."""
    preview = chunk["chunk_text"][:200].replace("\n", " ")
    return (
        f"[{rank}] {chunk.get('title', '?')} "
        f"(chunk_id={chunk['chunk_id']}, "
        f"score={chunk['score']:.3f})\n"
        f"    {preview}"
    )


def label_question(question_id: str, question: str,
                    candidates: list[dict]) -> dict:
    raw = input("Enter relevant candidate numbers: ")
    picks = [p.strip() for p in raw.split(",") if p.strip()]

    valid_picks = []
    for p in picks:
        if not p.isdigit() or not (1 <= int(p) <= len(candidates)):
            print(f"  skipping invalid entry: {p!r}")
            continue
        valid_picks.append(int(p))

    chunk_ids = [candidates[p - 1]["chunk_id"] for p in valid_picks]

    return {
        "question_id": question_id,
        "question": question,
        "relevant_chunk_ids": chunk_ids,
    }

def run() -> list[dict]:
    """Label every sample question, return records to save."""
    retriever = Retriever()
    result_list = []

    for id, question in enumerate(SAMPLE_QUESTIONS):
        records = retriever.search(question, top_k=TOP_K)
        result_list.append(label_question(f"q{id:02d}", question, records))    
    return result_list


if __name__ == "__main__":
    records = run()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} labeled questions to {OUT_PATH}")