"""Pipeline state and per-step trace records for the agentic
LegalEval workflow. PipelineState is a mutable accumulator;
StepLog is a frozen completed record."""

from dataclasses import dataclass, field
from typing import Literal
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "Evaluation")
)
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "Generation")
)
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "Retrieval")
)

from retriever import Retriever
from model_client import ModelClient
from generator import GeneratedAnswer
from claim_extractor import Claim
from report_generator import EvalReport
from retrieval_evaluator import RelevanceJudgment
from citation_checker import ClaimVerdict, load_chunk_lookup

TERMINAL_STATUSES = {'completed', 'abstained',
                     'failed', 'completed_with_warning'}

@dataclass(frozen=True)
class StepLog:
    """One completed step in the trace. Frozen — a finished
    record, never mutated after the orchestrator builds it."""
    step_name: str
    status: Literal["ok", "failed", "skipped", "warning"]
    latency_ms: float
    input_summary: str
    output_summary: str
    tokens: (int | None) = None


@dataclass
class PipelineState:
    """Mutable accumulator threaded through every step."""

    # --- Inputs (set once) ---
    client: ModelClient
    retriever: Retriever
    question: str
    question_id: str
    top_k: int
    strict_citations: bool = False
    correction_budget: int = 1

    # --- Artifacts (start empty/None, one per step) ---
    chunks: list[dict] = field(default_factory=list)
    answer: GeneratedAnswer|None = None
    claims: list[Claim] = field(default_factory=list)
    verdicts: list[ClaimVerdict]|None = field(default_factory=list)
    report: EvalReport|None = None
    relevance: RelevanceJudgment|None = None
    lookup: dict[str,str] = field(default_factory=load_chunk_lookup)

    # --- Trace ---
    steps: list[StepLog] = field(default_factory=list)

    # --- Control ---
    status: str = "running"
    correction_attempts: int = 0

    def record(self, log:StepLog) -> None:
        self.steps.append(log)
    
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES