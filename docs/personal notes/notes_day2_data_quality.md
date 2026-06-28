# Day 2 Theory — Data Quality for RAG

Why chunk size, overlap, metadata, and document cleanliness affect
retrieval quality. Notes tied to the LegalEval ingestion build.


## 1. Chunk size

**Takeaway:** Chunk size sets the retrieval unit — too big dilutes the
embedding signal, too small fragments meaning.

One chunk becomes one fixed-length vector that must summarize the whole
chunk. A large chunk averages many sub-topics into one vector, muting its
similarity to a focused query and feeding the LLM irrelevant text. A tiny
chunk embeds cleanly but can split a rule from its exception or cite.
300–500 words holds one coherent idea while keeping the vector sharp.

Build: 300 words sits at the precision end. Day 8 ablation (300/top_k 5
vs 500/top_k 8) measures whether 500 buys more groundedness.


## 2. Overlap

**Takeaway:** Overlap is insurance against a relevant span being split
across a chunk boundary.

Sliding-window cuts at fixed word counts, blind to meaning, so a key
sentence can land half in chunk N and half in N+1 — neither embeds the
full thought. Carrying ~50 words forward keeps any short span intact
somewhere. Cost: duplication inflates the index and risks near-duplicate
retrieval, where two adjacent chunks waste a top-k slot on one query.

Build: 50/300 is ~17%, a standard ratio. Near-duplicate cost interacts
with top_k — watch for overlapping neighbors crowding top-k in Day 5.


## 3. Metadata

**Takeaway:** Metadata turns a retrieved chunk into a citable,
filterable, auditable source — the backbone of an evaluation-first
system.

chunk_text buys the semantic match; metadata buys everything after:
citation (title + URL + section), filtering (jurisdiction, topic),
provenance for auditability, and ranking signals (recency, authority).
In law a superseded statute or wrong-circuit opinion is actively
misleading, so date and jurisdiction earn their keep.

Build: the sources.csv join in ingest.py is what makes Day 5 citation
coverage computable at all.


## 4. Document cleanliness

**Takeaway:** Noise in chunk_text corrupts both the embedding and the
generation — dirty text degrades retrieval silently.

Two failure modes. Retrieval: nav, banners, repeated headers/footers,
and page-number lines embed into the vector, pulling it off-topic so
real chunks get buried. Generation/citation: boilerplate and mangled
text (ff-ligature "Di!erent", missed de-hyphenation, footnotes folded
into body) feed the LLM garbage it may quote and break a citation
checker matching against source text. This is why naive tag-stripping
fails — target the content region instead.

Build: matches the accepted limitation set. Footnotes in body text can
create spurious "supporting" matches; ligature mangling can break
exact-quote citation checks in Day 5. Scoped and fine for v1 — as long
as the cost is stated.

Adjacent: token_count is a word-count proxy. It under-counts real
tokens (dense with "§", cites, hyphenated terms), so context-budget
math (Day 4) and cost math (Day 8) run optimistic. Cheap fix later:
swap in the real tokenizer.
