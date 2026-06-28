# Day 1 — RAG Fundamentals (Study Notes)

Personal notes for the LegalEval project. Each section: takeaway + short
explanation.

---

## Chunking — key parameters

- **Chunk size** (300-500 words): too small loses context; too large
  dilutes the answer and wastes context-window space.
- **Overlap** (50-100 words): consecutive chunks share boundary text so an
  idea split across a boundary isn't lost. Cheap insurance.
- **Project link:** bad splits hurt citation grounding (an answer could
  cite a source containing only half a rule). Day 8 tests chunk size
  300 vs 500 for this reason.

---

## Embeddings — how they're generated

- The embedding model was **trained on enormous amounts of text**, learning
  which words appear in similar contexts.
- It converts text into a **vector** (a list of numbers, e.g. 384) where
  similar *meaning* produces similar numbers. Meaning becomes proximity in
  space.
- **Example** (simplified to 3 dimensions = [employment, negative-action,
  money]):
      "termination" -> [0.91, 0.85, 0.20]
      "firing"      -> [0.90, 0.88, 0.15]   (almost identical to above)
      "severance"   -> [0.88, 0.30, 0.95]   (employment, but money-leaning)
      "banana"      -> [0.02, 0.10, 0.05]   (unrelated)
- Works on **whole chunks**, not just words: "terminated without cause" and
  "fired without a valid reason" get near-identical vectors despite sharing
  no words.
- **Caveat:** real dimensions aren't human-readable; the
  [employment/money] labels are a teaching simplification.

---

## Vector search — brute force vs ANN

- **Brute force (exact search):** compare the question vector against every
  chunk, sort by similarity. Fine for hundreds of vectors, but it's a
  linear scan — too slow at scale (millions of vectors = millions of
  comparisons per query).
- **ANN (Approximate Nearest Neighbor):** pre-organizes vectors into a
  graph/clusters so search jumps to the right neighborhood and checks only
  a fraction. "Approximate" = may rarely miss the true closest match, but
  vastly faster.
- **Why we use it:** exact = perfect but slow at scale; ANN = near-perfect
  but fast. The trade-off is almost always worth it.
- FAISS / ChromaDB handle storage + fast search. At this project's scale
  (a few hundred chunks) either is overkill on speed, so pick for
  convenience — ChromaDB is the gentler start.

---

## Top-k retrieval

- Top-k = how many of the top-ranked chunks you pull back to feed the
  answer generator. k=5 means "give me the 5 most relevant chunks."
- **Too low** (1-2): may miss the chunk holding the answer; model answers
  from incomplete context or guesses (hallucinates).
- **Too high** (15+): floods context with marginal chunks, dilutes signal,
  costs tokens, muddies citation grounding.
- No universal right k — depends on chunk size, question breadth, and
  noise/cost tolerance. That's *why* it's a Day 8 experiment variable
  (300/k5 vs 500/k8), tuned against metrics, not assumed.

---

## Context windows

- A model can only see a fixed amount of text at once (its context window,
  in tokens). Prompt + retrieved chunks + answer all share one budget.
- Tokens ~= 3/4 of a word (100 tokens ~= 75 words).
- **Chunk-size / top-k / context-window are a single connected system:**
  the three share one token budget, so tuning one affects the others.
  Important because Day 8's experiments are really about budgeting this
  space well — you can't change k without changing how much context fits.
- **Retrieval exists precisely to send less, but more relevant, text — not
  to fill the window. Good RAG is about precision, not volume.** Big
  windows tempt "stuff everything in," but more context costs money,
  latency, and reliability ("lost in the middle" — models attend less well
  to info buried in long contexts).

---

## Grounded generation

- The model answers **from the retrieved chunks, not its own memory**, with
  every claim traceable to a provided source. Enforced via prompt
  constraints, structured output, and **abstention** when context is
  insufficient.
- Not automatic: LLMs default to *fluent*, answering from trained-in memory
  unless actively constrained. Grounding is what turns "fluent" into
  "trustworthy" — non-negotiable in legal AI because people act on the
  answers.
- **Two-act structure:** generation grounds (Day 4); the Citation Verifier
  + Evaluator *check* whether grounding actually succeeded (Day 5). This is
  why evaluation is the project's primary feature.
