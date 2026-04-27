from rag.embedder import embed_query
from rag.vectorstore import search


def retrieve(question: str, n_results: int = 5) -> list[dict]:
    """Return top-n relevant chunks for the given question."""
    query_embedding = embed_query(question)
    results = search(query_embedding, n_results=n_results)

    if not results['documents'] or not results['documents'][0]:
        return []

    return [
        {
            'text': doc,
            'metadata': meta,
            'score': 1.0 - dist,  # cosine distance → similarity
        }
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
        )
    ]
