import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "Retrieval")
)
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "Generation")
)

from retriever import Retriever
from model_client import get_model_client
from generator import generate

from claim_extractor import extract_claims
from citation_checker import check_claims, load_chunk_lookup
from retrieval_evaluator import load_golden, evaluate_retrieval, judge_relevance
from report_generator import build_report

retriever = Retriever()
client = get_model_client("anthropic", "claude-opus-4-6")

QUESTION_ID = "q00"
QUESTION = "Can my employer fire me for being pregnant?"
VALUE_K = 5

result_retrieved = retriever.search(QUESTION, top_k = VALUE_K)
answer = generate(QUESTION, result_retrieved, client)
claims = extract_claims(answer.answer)
chunk_lookup = load_chunk_lookup()
claims_result = check_claims(claims, answer.citations, chunk_lookup, client)
golden = load_golden()
golden_q00 = [g for g in golden if g["question_id"] == QUESTION_ID]
retrieval_result = evaluate_retrieval(golden_q00, retriever, VALUE_K)[0]
report = build_report(QUESTION_ID, claims_result, retrieval_result)
print(report)

### OR ###

judgment = judge_relevance(
    QUESTION_ID, QUESTION, result_retrieved, client,
)
print(judgment)