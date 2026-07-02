import json
from pathlib import Path

import faiss

from src.Retrieval.embedder import Embedder

INDEX_PATH = Path("data/processed/faiss.index")
RECORDS_PATH = Path("data/processed/index_records.jsonl")


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.index = faiss.read_index(str(INDEX_PATH))
        self.records = self._load_records(RECORDS_PATH)

    @staticmethod
    def _load_records(path: Path) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = self.embedder.embed([query])
        scores, ids = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            hit = dict(self.records[idx])
            hit["score"] = float(score)
            results.append(hit)
        return results


def print_results(query: str, results: list[dict]) -> None:
    """Pretty-print retrieval hits: title, source, score, preview."""
    print(f"\nQUERY: {query}")
    for rank, r in enumerate(results, 1):
        preview = r["chunk_text"][:160].replace("\n", " ")
        print(f"\n[{rank}] score={r['score']:.3f}")
        print(f"    title:  {r.get('title', '?')}")
        print(f"    source: {r.get('source_url', '?')}")
        print(f"    text:   {preview}...")


if __name__ == "__main__":
    retriever = Retriever()

    query = "Can my employer fire me for being pregnant?"
    results = retriever.search(query, top_k=5)
    print_results(query, results)