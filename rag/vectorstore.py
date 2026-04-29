from __future__ import annotations

import os
from pathlib import Path
import chromadb

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        db_path = os.getenv('VECTORDB_PATH', 'vectordb')
        Path(db_path).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=db_path)
        collection_name = os.getenv('COLLECTION_NAME', 'normativos')
        _collection = _client.get_or_create_collection(
            name=collection_name,
            metadata={'hnsw:space': 'cosine'},
        )
    return _collection


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    collection = get_collection()
    collection.upsert(
        ids=[c['chunk_id'] for c in chunks],
        documents=[c['text'] for c in chunks],
        embeddings=embeddings,
        metadatas=[c['metadata'] for c in chunks],
    )


def search(query_embedding: list[float], n_results: int = 5) -> dict:
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances'],
    )


def get_by_ids(ids: list[str]) -> list[dict]:
    """Fetch specific chunks by their chunk_id."""
    if not ids:
        return []
    collection = get_collection()
    results = collection.get(ids=ids, include=['documents', 'metadatas'])
    if not results['documents']:
        return []
    return [
        {'text': doc, 'metadata': meta, 'score': None}
        for doc, meta in zip(results['documents'], results['metadatas'])
    ]


def is_indexed(filename: str) -> bool:
    """Return True if this file already has chunks in the collection."""
    collection = get_collection()
    results = collection.get(
        where={"source_file": filename},
        limit=1,
        include=["metadatas"],
    )
    return len(results.get("ids", [])) > 0


def count() -> int:
    return get_collection().count()
