import json
from dataclasses import dataclass

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

QUESTION_ID = "q00"
QUESTION = "Can my employer fire me for being pregnant?"

if __name__ == "__main__":
    judgment = judge_relevance(
        QUESTION_ID, QUESTION, result_retrieved, client,
    )
    print(judgment)