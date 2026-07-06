"""Streamlit UI for LegalEval — Day 7 dashboard."""
from pathlib import Path
import sys
import uuid

import streamlit as st

# --- Pipeline import boundary ------------------------------------
# The ONLY import from src. Isolates folder-casing to this one
# line; Day 9 cleanup touches just this path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "Agent"))
from orchestrator import run_pipeline  # noqa: E402
from state import PipelineState

# state.py inserts src/Retrieval and src/Generation onto sys.path at
# import time, so these are safe to import here after the above.
from retriever import Retriever          # noqa: E402
from model_client import get_model_client  # noqa: E402


@st.cache_resource
def load_resources():
    """Load FAISS index + embedder ONCE per session."""
    return Retriever()


def sidebar_controls():
    st.sidebar.header("Configuration")
    top_k = st.sidebar.slider("Top-k retrieved", 1, 10, 5)
    model = st.sidebar.selectbox(
        "Model",
        [
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-haiku-4-5-20251001",
        ],
    )
    return top_k, model


def render_answer(state):
    st.subheader("Answer")
    answer = state.answer

    if not answer or not answer.answer:
        st.warning("No answer generated — pipeline abstained.")
        if answer and answer.limitations:
            st.caption(answer.limitations)
        return

    if answer.needs_human_review:
        st.warning("Confidence low — human review recommended.")
    if state.status == "completed_with_warning":
        st.warning("High hallucination risk remains after correction.")

    st.write(answer.answer)

    if answer.citations:
        st.markdown("**Citations:** " + " · ".join(f"`{c}`" for c in answer.citations))

    if answer.supporting_quotes:
        with st.expander("Supporting quotes"):
            for q in answer.supporting_quotes:
                st.markdown(f"> {q}")

    if answer.limitations:
        st.info(answer.limitations)


def render_sources(state):
    if not state.chunks:
        return
    st.subheader("Retrieved Sources")
    for chunk in state.chunks:
        title = chunk.get("title") or chunk.get("chunk_id", "Source")
        with st.expander(title):
            score = chunk.get("score")
            if score is not None:
                st.caption(f"Score: {score:.3f}")
            url = chunk.get("source_url")
            if url:
                st.markdown(f"[{url}]({url})")
            preview = chunk.get("chunk_text", "")[:300]
            st.markdown(preview)


def render_eval_panel(state):
    if not state.report:
        return
    st.subheader("Evaluation")
    report = state.report

    total_latency = sum(s.latency_ms for s in state.steps)
    relevance_score = state.relevance.score if state.relevance else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Citation Coverage", f"{report.citation_coverage:.0%}")
    col2.metric("Hallucination Risk", report.hallucination_risk.capitalize())
    col3.metric("Retrieval Relevance", f"{relevance_score}/5" if relevance_score else "—")
    col4.metric("Total Latency", f"{total_latency:.0f} ms")

    verdicts = state.verdicts or []
    if verdicts:
        st.markdown("**Groundedness**")
        rows = [
            {"Claim": v.claim_text, "Verdict": v.verdict, "Reasoning": v.reasoning}
            for v in verdicts
        ]
        st.dataframe(rows, use_container_width=True)


def render_trace(state):
    if not state.steps:
        return
    st.subheader("Agent Trace")
    STATUS_ICON = {"ok": "✅", "failed": "❌", "skipped": "⏭", "warning": "⚠️"}
    rows = [
        {
            "Step": s.step_name,
            "Status": f"{STATUS_ICON.get(s.status, '')} {s.status}",
            "Latency (ms)": f"{s.latency_ms:.0f}",
            "Tokens": s.tokens if s.tokens is not None else "—",
            "Detail": s.output_summary
        }
        for s in state.steps
    ]
    st.dataframe(rows, use_container_width=True)


def main():
    st.set_page_config(page_title="LegalEval", layout="wide")
    st.title("LegalEval — Legal RAG Evaluation Workbench")
    st.caption(
        "Personal educational prototype. Not legal advice. "
        "Not affiliated with LexisNexis."
    )

    top_k, model = sidebar_controls()
    question = st.text_area("Legal question", height=100)
    run = st.button("Run", type="primary")

    if run and question.strip():
        with st.spinner("Running pipeline..."):
            try:
                retriever = load_resources()
                client = get_model_client("anthropic", model)
                state = PipelineState(
                    client=client,
                    retriever=retriever,
                    question=question,
                    question_id=str(uuid.uuid4()),
                    top_k=top_k,
                )
                state = run_pipeline(state)
                st.session_state["state"] = state
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.session_state.pop("state", None)

    state = st.session_state.get("state")
    if state is not None:
        render_answer(state)
        render_sources(state)
        render_eval_panel(state)
        render_trace(state)


if __name__ == "__main__":
    main()
