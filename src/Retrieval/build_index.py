import json
from pathlib import Path

import faiss
import numpy as np

from src.Retrieval.embedder import Embedder

# Paths are relative to the project root (run from there).
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/faiss.index")
RECORDS_PATH = Path("data/processed/index_records.jsonl")


def load_chunks(path: Path) -> list[dict]:
    """Read chunks.jsonl into a list of dicts, preserving file order."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_index(records: list[dict]) -> faiss.Index:

    embedder = Embedder()
    texts = [r["chunk_text"] for r in records]

    vectors = embedder.embed(texts)
    vectors = np.ascontiguousarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(embedder.dim)

    index.add(vectors)
    return index



def persist(index: faiss.Index, records: list[dict]) -> None:
    """Write the index and the order-aligned records to disk."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with open(RECORDS_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    records = load_chunks(CHUNKS_PATH)
    print(f"loaded {len(records)} chunks")

    index = build_index(records)
    print(f"index has {index.ntotal} vectors, dim {index.d}")

    persist(index, records)
    print(f"wrote {INDEX_PATH} and {RECORDS_PATH}")