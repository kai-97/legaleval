

def chunk_text(text:str, doc_meta:dict, chunk_size:int, overlap:int) -> list:
    words = text.split()
    total_words = len(words)
    iterator = 0
    chunks = []

    index = 0
    while iterator < total_words:
        chunk = " ".join(words[iterator:iterator+chunk_size])
        chunks.append({
            "chunk_id": f"{doc_meta["document_id"]}_{index}",
            "document_id": doc_meta["document_id"],
            "title": doc_meta.get("title"),
            "source_url": doc_meta.get("source_url"),
            "jurisdiction": doc_meta.get("jurisdiction"),
            "date": doc_meta.get("date"),
            "section": doc_meta.get("section"),
            "chunk_text": chunk,
            "token_count": len(chunk.split())
        })
        index += 1
        if iterator + chunk_size >= total_words:
            break
        iterator += (chunk_size - overlap)
        
    return chunks


if __name__ == "__main__":
    text = "test text"
    file = chunk_text(text, {}, 300, 50)
    print(len(file[0]))