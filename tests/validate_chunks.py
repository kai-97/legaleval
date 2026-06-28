
import json

def validate_chunks(path="data/processed/chunks.jsonl"):
    with open(path, encoding="utf-8") as f:
        passed, failed, warning = 0,0,0
        failed_chunk_ids = []
        warning_ids = []
        for line in f:
            chunk = json.loads(line)

            # Empty chunk_text — chunk_text blank or whitespace-only → fail
            if not chunk['chunk_text']:
                print(f"{chunk['chunk_id']} has no chunk text")
                failed += 1
                failed_chunk_ids.append(chunk['chunk_id'])
                continue
            
            # Tiny chunks — token_count < 20 → warning (likely junk tail)
            if chunk['token_count'] < 20:
                print(f"WARN: {chunk['chunk_id']} is too short.")
                warning += 1
                warning_ids.append(chunk['chunk_id'])

            # Missing required metadata — chunk_id, document_id, title, source_url, jurisdiction present and non-empty → fail if missing (note: date/section are allowed blank/null, so treat those as warnings, not failures)
            for parameter in ['chunk_id', 'document_id', 'title', 'source_url', 'jurisdiction']:
                if not chunk[parameter]:
                    print(f"{chunk['chunk_id']} has no {parameter}")
                    failed += 1
                    failed_chunk_ids.append(chunk['chunk_id'])
                    continue
            
            if chunk['chunk_id'] not in warning_ids:
                passed += 1
        # Summary — total chunks, failures, warnings, and which chunk_ids
        print(f"Passed: {passed}\nFailed: {failed}\nFailed Chunks: {failed_chunk_ids}\nWarning: {warning}\nWarning Chunks: {warning_ids}")

if __name__ == "__main__":
    validate_chunks()
