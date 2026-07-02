# scratch, don't commit
from src.Retrieval.retriever import Retriever
from src.Generation.model_client import get_model_client
from src.Generation.generator import generate, build_context, SYSTEM_PROMPT



q = "What makes a non-compete enforceable?"
chunks = Retriever().search(q, top_k=3)      # instantiate, then call
client = get_model_client("anthropic", "claude-opus-4-6")

resp = client.complete(system=SYSTEM_PROMPT, user=build_context(q, chunks))
print(repr(resp.text))     # repr shows fences/whitespace literally


# ans = generate(q, chunks, client)
# print(ans)