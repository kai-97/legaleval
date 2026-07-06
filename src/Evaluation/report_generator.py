from dataclasses import dataclass
from citation_checker import ClaimVerdict

@dataclass(frozen=True)
class EvalReport:
    question_id: str
    citation_coverage: float
    supported_count: int
    partially_supported_count: int
    unsupported_count: int
    hallucination_risk: str  # "low" | "medium" | "high"
    retrieval_precision: float | None
    retrieval_recall: float | None
    retrieval_mrr: float | None


def summarize_verdicts(verdicts: list[ClaimVerdict]) -> dict:
    """Count claims by verdict, compute citation_coverage."""
    if len(verdicts) == 0:
        raise ValueError("No Verdicts here.")
    
    sup, par, uns, err = 0,0,0, 0

    for verdict in verdicts:
        if verdict.verdict == "supported":
            sup += 1
        elif verdict.verdict == "partially_supported":
            par += 1
        elif verdict.verdict == "parse_error":
            err += 1
        else:
            uns += 1
    
    scored = sup + par + uns
    coverage = (sup+par)/scored if scored else 0.0
    
    return {"supported": sup,
            "partially_supported": par,
            "unsupported": uns,
            "parse_error": err,
            "citation_coverage": coverage}


def classify_hallucination_risk(
    unsupported_count: int, total_claims: int,
) -> str:
    if not total_claims:
        raise ValueError("No total claims.")
    
    percentage = unsupported_count*100/total_claims

    if percentage < 30.0:
        return "low"
    elif 30 <= percentage < 70:
        return "medium"
    else:
        return "high"
    

def build_report(
    question_id: str,
    verdicts: list[ClaimVerdict],
    retrieval_result: dict | None = None,
) -> EvalReport:
    result_dict = summarize_verdicts(verdicts)
    hallu_risk = classify_hallucination_risk(result_dict["unsupported"], len(verdicts))
    precision = retrieval_result["precision"] if retrieval_result else None
    recall = retrieval_result["recall"] if retrieval_result else None
    rr = retrieval_result["reciprocal_rank"] if retrieval_result else None
    return EvalReport(
        question_id, result_dict["citation_coverage"],
        result_dict["supported"], result_dict["partially_supported"],
        result_dict["unsupported"], hallu_risk, precision, recall, rr)
