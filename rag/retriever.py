import os
from rag.embedder import embed_query
from rag.vectorstore import search, get_by_ids


def _neighbor_ids(chunks: list[dict], window: int) -> list[str]:
    """Compute chunk IDs of neighbors within ±window positions."""
    ids = []
    for chunk in chunks:
        meta = chunk['metadata']
        filename = meta.get('source_file', '')
        pos = meta.get('chunk_position')
        total = meta.get('total_chunks', 0)
        if pos is None:
            continue
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            neighbor_pos = pos + delta
            if 0 <= neighbor_pos < total:
                ids.append(f"{filename}_chunk_{neighbor_pos:04d}")
    return ids


def retrieve(question: str, n_results: int = 5) -> list[dict]:
    """
    Retrieve top-n relevant chunks by semantic similarity, then expand
    each result with its adjacent chunks (context_window on each side).
    The final list is sorted by document and position so the LLM reads
    coherent passages in order.
    """
    context_window = int(os.getenv('CONTEXT_WINDOW', '1'))

    query_embedding = embed_query(question)
    results = search(query_embedding, n_results=n_results)

    if not results['documents'] or not results['documents'][0]:
        return []

    matched = [
        {'text': doc, 'metadata': meta, 'score': 1.0 - dist}
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
        )
    ]

    if context_window > 0:
        existing_ids = {c['metadata']['chunk_id'] for c in matched}
        ids_to_fetch = [
            cid for cid in _neighbor_ids(matched, context_window)
            if cid not in existing_ids
        ]
        neighbors = get_by_ids(list(set(ids_to_fetch)))
        matched = matched + neighbors

    # Deduplicate and sort by (document, position) for coherent reading order
    seen: set[str] = set()
    unique: list[dict] = []
    for chunk in matched:
        cid = chunk['metadata']['chunk_id']
        if cid not in seen:
            seen.add(cid)
            unique.append(chunk)

    unique.sort(key=lambda c: (
        c['metadata'].get('source_file', ''),
        c['metadata'].get('chunk_position', 0),
    ))

    return unique
