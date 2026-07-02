from dataclasses import dataclass
import json

@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    limitations: str
    citations: list[str]
    supporting_quotes: list[str]
    needs_human_review: bool

    @classmethod
    def abstention(cls, reason: str) -> "GeneratedAnswer":

        return cls(
            answer = "",
            limitations=reason,
            citations=[],
            supporting_quotes=[],
            needs_human_review=True
    )

SYSTEM_PROMPT = """\
You are a grounded legal-research assistant for US employment
law. You answer ONLY from the provided source chunks.

grounding rules
   - use only the chunks; never outside knowledge
   - cite the chunk_id of every chunk you draw from
   - this is not legal advice (project disclaimer) so do not bluff

abstention rule
   - if chunks don't support a confident answer, set
     needs_human_review true, leave answer empty
   - do not guess or fill gaps

output contract — JSON ONLY, no prose, no ``` fences
Response with a single raw JSON object where 
the first character of the JSON should be a '{' and the last '}'.
Do not use markdown code fences or any text before or after the JSON.
Example only for format: {"answer": "...", ...}
   keys: answer (str), citations (list of chunk_id),
   supporting_quotes (list of str, verbatim from chunks),
   limitations (str), needs_human_review (bool)
"""

def generate(question: str, chunks: list[dict], client) -> GeneratedAnswer:
    if not chunks:
        return GeneratedAnswer.abstention("No Chunks")
    

    resp = client.complete(user=build_context(question, chunks),
                        system=SYSTEM_PROMPT)
    try:
        data = json.loads(resp.text)
        return GeneratedAnswer(**data)
    except (json.JSONDecodeError, TypeError):
        return GeneratedAnswer.abstention("JSON returned incorrectly")

def build_context(question:str, chunks: list[dict]):
    USER_PROMPT = f"""
    Here is the question that the user asked: {question}.

    Below are the relevant chunks that might be able to support the
    question the user has asked. Use that as a reference and make a note
    of the citations you make.
    """

    for chunk in chunks:
        USER_PROMPT += f"[{chunk["chunk_id"]}] {chunk["chunk_text"]}"
    
    return USER_PROMPT