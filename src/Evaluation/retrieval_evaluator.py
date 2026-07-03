import json
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "Retrieval")
)
from retriever import Retriever

GOLDEN_PATH = Path("data/eval/golden_relevance.jsonl")


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    """Load golden relevance records (question_id, question,
    relevant_chunk_ids) from the labeled jsonl."""

    result = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            result.append(json.loads(line.strip()))
    return result


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int,
) -> float:
    if k <= 0:
        raise ValueError("k should be > 0")
    
    val = 0
    for id in retrieved_ids[:k]:
        if id in relevant_ids:
            val += 1
    return val/k


def recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int,
) -> float:
        if k <= 0:
            raise ValueError("k should be > 0")
        if not relevant_ids:
            raise ValueError("No relevant chunks here")
        
        val = 0
        for id in retrieved_ids[:k]:
            if id in relevant_ids:
                val += 1
        return val/len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str], relevant_ids: set[str],
) -> float:
    """1 / rank of the first relevant chunk in retrieved_ids,
    or 0.0 if none of them are relevant."""
    for rank, id in enumerate(retrieved_ids, start = 1):
        if id in relevant_ids:
            return 1/rank
    return 0.0


def evaluate_retrieval(
    golden: list[dict], retriever: Retriever, k: int,
) -> list[dict]:
    result = []
    for record in golden:
        ret_results = retriever.search(record["question"], top_k=k)
        retrieved_ids = [ret["chunk_id"] for ret in ret_results]
        relevant_ids = set(record["relevant_chunk_ids"])
        result.append({"question_id":record["question_id"],
                       "precision": precision_at_k(retrieved_ids, relevant_ids, k),
                       "recall": recall_at_k(retrieved_ids, relevant_ids, k),
                       "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids)})
    return result


RELEVANCE_SYSTEM_PROMPT = """You are judging retrieval quality
for a legal question-answering system. You will be given a
question and a list of retrieved passages, in the order they
were returned (rank 1 first).

Score how well the ranking surfaces relevant passages early,
from 1 to 5:
- 5: the most relevant passages appear at or near the top; no
  clearly irrelevant passage outranks a clearly relevant one.
- 3: a mix — some relevant passages present, but ranking is
  inconsistent (relevant passages buried below irrelevant ones,
  or ranking is patchy).
- 1: retrieved passages are irrelevant to the question, or the
  few relevant ones (if any) are ranked at or near the bottom.

Judge ranking quality specifically — not whether every passage
is individually relevant, but whether the ordering puts the
best material first.

Respond with a single raw JSON object. First character must be
{ and last character must be }. No markdown fences, no prose
outside the object.

{"score": 5, "reasoning": "one sentence explaining the ranking
quality"}
"""

@dataclass(frozen=True)
class RelevanceJudgment:
    question_id: str
    score: int  # 1-5
    reasoning: str

def format_ranked_chunk(rank: int, chunk: dict) -> str:
    """Render one retrieved chunk for the judge prompt."""
    preview = chunk["chunk_text"][:200].replace("\n", " ")
    return (
        f"[{rank}] (chunk_id={chunk['chunk_id']})\n"
        f"    {preview}"
    )

def judge_relevance(
    question_id: str,
    question: str,
    retrieved_chunks: list[dict],
    client,
) -> RelevanceJudgment:
    """Ask the model to score ranking quality 1-5."""

    listing = "\n\n".join(
        format_ranked_chunk(i,chunk)
        for i, chunk in enumerate(retrieved_chunks, start=1)
        )
    user_prompt = f"Question: {question}\n\n{listing}"
    try:
        raw = client.complete(
            system=RELEVANCE_SYSTEM_PROMPT, user=user_prompt,
        ).text
        parsed = json.loads(raw)
        score = parsed["score"]
        if not isinstance(score, int) or not (1 <= score <= 5):
            raise ValueError(f"unexpected score: {score}")
        return RelevanceJudgment(
            question_id, score, parsed["reasoning"],
        )
    except (
        json.JSONDecodeError, KeyError, ValueError,
    ) as e:
        return RelevanceJudgment(
            question_id, score=1,
            reasoning=f"judge parse failure: {e}",
        )
