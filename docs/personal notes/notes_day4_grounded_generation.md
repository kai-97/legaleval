# Day 4 — Grounded Generation (Theory)

1. Hallucination — fluent, confident claims not supported by any
   source; the core risk in legal AI.

2. Unsupported claims — the measurable form of hallucination: a
   claim with no backing in retrieved context.

3. Abstention — declining to answer when support is insufficient,
   rather than guessing (code-level guard + prompt-level flag).

4. Citation grounding — every claim traceable to a specific source
   via chunk_id + verbatim supporting quotes.

5. Structured outputs — forcing a fixed schema (five-key JSON) so
   downstream code can consume output deterministically.

6. Prompt constraints — instructions that bound behavior
   (only-from-chunks, verbatim quotes, JSON-only) as guardrails
   against specific failure modes.
