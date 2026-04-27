def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into word-based chunks with overlap."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(' '.join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


def chunk_document(
    segments: list[dict],
    filename: str,
    file_type: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Receive parsed segments and return a flat list of chunks with metadata.
    Each segment is {'text': str, 'page': int|None}.
    """
    chunks = []
    chunk_idx = 0

    for segment in segments:
        text_chunks = split_into_chunks(segment['text'], chunk_size, overlap)
        for chunk_text in text_chunks:
            if not chunk_text.strip():
                continue

            chunk_id = f"{filename}_chunk_{chunk_idx:04d}"
            metadata: dict = {
                'source_file': filename,
                'file_type': file_type,
                'chunk_id': chunk_id,
            }
            if segment.get('page') is not None:
                metadata['page'] = segment['page']

            chunks.append({
                'text': chunk_text,
                'metadata': metadata,
                'chunk_id': chunk_id,
            })
            chunk_idx += 1

    return chunks
