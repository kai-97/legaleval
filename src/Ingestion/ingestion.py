from pathlib import Path
import csv
import json

from loader import load_html, load_pdf
from cleaner import clean_html, clean_pdf
from chunker import chunk_text

def main() -> None:
    chunk_list = []
    sources = load_sources()
    for path in Path("data/raw").iterdir():
        file_dict = {}
        if path.suffix == ".pdf":
            file_dict = load_pdf(path.name)
            file_dict["body"] = clean_pdf(file_dict["body"])
        elif path.suffix == ".html":
            file_dict = load_html(path.name)
            file_dict["body"] = clean_html(file_dict["body"])
        else:
            print(f"{path.name} extension not supported")
            continue

        if path.name not in sources:
            print(f"No metadata for {path.name}, skipping")
            continue
        src = sources[path.name]
        doc_meta = {
            "document_id": path.stem,
            "title": file_dict["heading"] or src["title"],
            "source_url": src["url"],
            "jurisdiction": src["jurisdiction"],
            "date": src.get("date", ""),
            "section": file_dict.get("heading", "")
        }

        chunk_list.extend(chunk_text(file_dict["body"], doc_meta, 300, 50))
    print(f"Total chunks: {len(chunk_list)}")

    out_path = Path("data/processed/chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunk_list:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunk_list)} chunks to {out_path}")


def load_sources(path="src/Ingestion/sources_1.csv"):
    with open(path, encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}

if __name__ == "__main__":
    main()
