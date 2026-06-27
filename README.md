# LegalEval — Legal RAG & Agent Evaluation Workbench

> A personal applied-AI project that runs a retrieval-augmented legal Q&A
> workflow and **measures whether its answers are trustworthy** — grounded,
> cited, and honest about their limits.

## Disclaimer

This is a personal educational prototype. It does **not** provide legal
advice, is **not** a substitute for a licensed attorney, and is **not
affiliated with or endorsed by any employer**. All source documents are
public legal materials.

## Problem statement

Legal AI systems are expected to produce answers that are not just fluent
but *trustworthy* — grounded in a verifiable source of truth. A confident
answer with no traceable source, or one that subtly misstates the law, is
worse than no answer at all, because in legal work the cost of being wrong
is borne by real people. LegalEval addresses this by treating evaluation as
the primary feature rather than an afterthought: it runs a
retrieval-augmented legal Q&A workflow and then measures whether each answer
is actually grounded in its retrieved source of truth — scoring citation
coverage, claim support, retrieval relevance, and hallucination risk — so
that trust becomes something you can *measure*, not just hope for.

## Domain

Public US employment law (wrongful termination, workplace discrimination,
and wage-and-hour), at the federal level plus selected states.

## Architecture

_Coming Day 1 — see [docs/architecture.md](docs/architecture.md)._

## Project structure

    app/        Streamlit UI (Day 7)
    data/       raw + processed corpus, chunks.jsonl
    docs/       architecture, evaluation methodology, examples, limitations
    notebooks/  exploration and retrieval demos
    src/        ingestion, retrieval, generation, evaluation, agent
    tests/      validation and unit tests

## Setup

_Coming Day 9 — full setup and run instructions._

## Evaluation methodology

_Coming Day 5 — see [docs/evaluation_methodology.md](docs/evaluation_methodology.md)._

## Limitations & future work

_Coming Day 9 — see [docs/limitations.md](docs/limitations.md)._

## Status

Day 1 — scope, repo, architecture, and corpus.
