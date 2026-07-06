import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from claim_extractor import Claim

Verdict = Literal[
    "supported", "partially_supported", "unsupported", "parse_error"
]

CHUNKS_PATH = Path("data/processed/chunks.jsonl")

SYSTEM_PROMPT = """
    You are checking whether a legal claim is
supported by cited source text. You will be given a claim and
one or more source passages.

Classify the claim as exactly one of:
- supported: the source fully backs the claim's assertion.
- partially_supported: the source backs part of the claim but
  not all of it (e.g. confirms one detail but not another, or
  supports a narrower version of a broader claim).
- unsupported: the source does not back the claim, even if it
  is topically related.

Respond with a single raw JSON object. First character must be
{ and last character must be }. No markdown fences, no prose
outside the object.

{"verdict": "supported", "reasoning": "one sentence explaining
why, citing what in the source confirms or fails to confirm
the claim"}
"""

@dataclass(frozen=True)
class ClaimVerdict:
    claim_id: str
    claim_text: str
    verdict: Verdict
    reasoning: str


def load_chunk_lookup(path: Path = CHUNKS_PATH,) -> dict[str, str]:
    chunk_data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            chunk_data[record["chunk_id"]] = record["chunk_text"]
    return chunk_data
    

def check_claim(
    claim: Claim, source_text: str, client,
) -> ClaimVerdict:
    try:
        raw = client.complete(
            system=SYSTEM_PROMPT,
            user=f"Claim: {claim.text}\n\nSource:{source_text}",
        ).text
        raw = raw[raw.find("{"):raw.rfind("}") + 1]
        parsed = json.loads(raw)
        verdict = parsed["verdict"]
        if verdict not in (
            "supported", "partially_supported", "unsupported",
        ):
            raise ValueError(f"unexpected verdict: {verdict}")
        return ClaimVerdict(
            claim.claim_id, claim.text, verdict,
            parsed["reasoning"],
        )
    except (
        json.JSONDecodeError, KeyError, ValueError,
    ) as e:
        return ClaimVerdict(
            claim.claim_id, claim.text, "parse_error",
            reasoning=f"judge response parse failure: {e}",
        )

def check_claims(
    claims: list[Claim],
    cited_chunk_ids: list[str],
    chunk_lookup: dict[str, str],
    client,
) -> list[ClaimVerdict]:
    """Run check_claim for every claim against cited chunks."""
    valid_ids = [
        cid for cid in cited_chunk_ids if cid in chunk_lookup
    ]
    result = []

    if not valid_ids:
        for claim in claims:
            result.append(ClaimVerdict(claim.claim_id, claim.text, "unsupported", reasoning="No Valid IDs"))
    else:
        source_text = "\n\n".join(
            chunk_lookup[cid] for cid in valid_ids
        )
        for claim in claims:
            result.append(check_claim(claim, source_text, client))
    return result