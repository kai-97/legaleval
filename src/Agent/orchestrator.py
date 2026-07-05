"""Orchestrator for the agentic LegalEval pipeline. Owns
control flow, timing, and step logging. Step functions read
state and write one artifact each; the executor wraps them
with timing + StepLog construction."""

import time
from state import PipelineState, StepLog
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "Retrieval")
)
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "Generation")
)

from generator import generate
from retrieval_evaluator import judge_relevance
from report_generator import build_report
from citation_checker import check_claims
from claim_extractor import extract_claims 

def _run_step(state, step_name, fn, input_summary):
    """Execute one step with timing + logging. fn mutates
    state and returns (output_summary, status, tokens).
    On exception: log failed, set state.status = 'failed'."""
    start = time.perf_counter()
    try:
        output_summary, status, tokens = fn(state)
    except Exception as e:
        output_summary, status, tokens = str(e), "failed", None
        state.status = "failed"
    
    latency_ms = (time.perf_counter() - start) * 1000
    step_log = StepLog(step_name=step_name, status=status, latency_ms=latency_ms,
                       input_summary=input_summary, output_summary=output_summary,
                       tokens=tokens)
    state.record(step_log)
    return status


# --- Step functions: each reads state, writes ONE artifact,
#     returns (output_summary, status, tokens) ---

def step_retrieve(state):
    chunks = state.retriever.search(state.question, state.top_k)
    state.chunks = chunks

    if not chunks:
        return ("No Chunks", "warning", None)
    return (f"{len(chunks)} chunks", "ok", None)
    

def step_generate(state):
    answer = generate(state.question, state.chunks,
                      state.client, strict=state.strict_citations)
    state.answer = answer

    if not answer.answer:
        return "abstained", "warning", None
    return f"{len(answer.citations)} citations", "ok", None


def step_extract(state):
    state.claims = extract_claims(state.answer.answer)
    return f"{len(state.claims)} claims", "ok", None


def step_check(state):
    state.verdicts = check_claims(state.claims, state.answer.citations, state.lookup,
                                  state.client)
    return f"{len(state.verdicts)} verdicts", "ok", None


def step_evaluate(state):
    state.relevance = judge_relevance(
        state.question_id, state.question, state.chunks,
        state.client
    )
    return f"relevance {state.relevance.score}/5", "ok", None


def step_report(state):
    state.report = build_report(state.question_id, state.verdicts)
    return state.report.hallucination_risk, "ok", None


# --- Orchestration ---

def _summarize_input(state: PipelineState, name: str) -> str:
    if name == "generate":
        return f"{len(state.chunks)} chunks"
    elif name == "extract":
        return f"answer is {len(state.answer.answer)} characters long"
    elif name == "check":
        return f"{len(state.claims)} claims"
    elif name == "evaluate":
        return f"{len(state.chunks)} chunks"
    elif name == "report":
        return f"{len(state.verdicts)} verdicts"


def run_pipeline(state: PipelineState) -> PipelineState:
    """Full flow with one correction loop. Retrieval runs
    once; correction re-enters at generate."""

    if _run_step(state, "retrieve", step_retrieve,
                 state.question[:60]) == "failed":
        return state
    if not state.chunks:
        state.status = "abstained"
        return state

    steps = [
        ("generate", step_generate),
        ("extract",  step_extract),
        ("check",    step_check),
        ("evaluate", step_evaluate),
        ("report",   step_report),
    ]

    while not state.is_terminal():
        for name, fn in steps:
            summary = _summarize_input(state, name)
            if _run_step(state, name, fn, summary) == "failed":
                return state          # status already "failed"

        risk = state.report.hallucination_risk
        if risk != "high":
            state.status = "completed"
        elif state.correction_attempts < state.correction_budget:
            state.strict_citations = True
            state.correction_attempts += 1
            # loop repeats -> regenerate onward
        else:
            state.status = "completed_with_warning"

    return state