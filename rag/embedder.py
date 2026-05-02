"""Gera embeddings de texto usando BAAI/bge-m3 via sentence-transformers."""

__author__ = "William Mendes"

from __future__ import annotations

import os
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
        print(f"  [embedder] Loading model: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, show_progress_bar=True).tolist()


def embed_query(text: str) -> list[float]:
    return get_model().encode(text).tolist()
