"""Day 6 smoke test: run the full agentic pipeline on one
question and inspect the trace + final report."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "Generation")
)
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "Retrieval")
)

from retriever import Retriever
from model_client import get_model_client
from state import PipelineState
from orchestrator import run_pipeline


def main():
    client = get_model_client("anthropic", "claude-opus-4-6")        # Day 4 factory
    retriever = Retriever()

    state = PipelineState(
        client=client,
        retriever=retriever,
        question="Can my employer fire me for being pregnant?",
        question_id="q00",
        top_k=5,
    )

    run_pipeline(state)

    print(f"\nfinal status: {state.status}")
    print(f"correction_attempts: {state.correction_attempts}\n")

    print("--- trace ---")
    for log in state.steps:
        print(f"Step Name: {log.step_name}\nStatus: {log.status}\nLatency(ms):{log.latency_ms}\nSummaries:\nInput:{log.input_summary}\nOutput:{log.output_summary}\n")

    print(f"answer len: {len(state.answer.answer)}")
    print(f"limitations: {state.answer.limitations}")

    if state.report:
        print("\n--- report ---")
        print(f"Hallucination risk:{state.report.hallucination_risk}\nCitation Coverage: {state.report.citation_coverage}")


if __name__ == "__main__":
    main()