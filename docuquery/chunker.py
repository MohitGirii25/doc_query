"""Split long text into small overlapping word chunks so retrieval stays precise."""


def chunk_text(text, size=60, overlap=15):
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + size])
        if piece:
            chunks.append(piece)
        if start + size >= len(words):
            break
    return chunks


def chunk_records(records, size=60, overlap=15):
    """Chunk a list of {'source', 'text'} records, keeping the source name."""
    out = []
    for rec in records:
        for piece in chunk_text(rec["text"], size, overlap):
            out.append({"source": rec["source"], "text": piece})
    return out
