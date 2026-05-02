"""Divide segmentos de documentos em chunks baseados em parágrafos com metadados completos."""

__author__ = "William Mendes"

from __future__ import annotations

import hashlib
import re


def _doc_id(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:8]


def _split_paragraphs(text: str, min_words: int = 80, max_words: int = 350) -> list[str]:
    """
    Divide o texto em parágrafos (linhas em branco como delimitadores), agrupa os
    pequenos e quebra os muito longos, preservando os limites de parágrafo.
    """
    raw = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for para in raw:
        word_count = len(para.split())

        if word_count > max_words:
            # Esvazia as partes acumuladas antes de continuar
            if current_parts:
                chunks.append('\n\n'.join(current_parts))
                current_parts = []
                current_words = 0
            # Quebra o parágrafo longo por contagem de palavras
            words = para.split()
            for i in range(0, len(words), max_words):
                chunks.append(' '.join(words[i:i + max_words]))

        elif current_words + word_count > max_words and current_words >= min_words:
            # Grupo atual cheio — salva e inicia um novo
            chunks.append('\n\n'.join(current_parts))
            current_parts = [para]
            current_words = word_count

        else:
            current_parts.append(para)
            current_words += word_count

    if current_parts:
        chunks.append('\n\n'.join(current_parts))

    return chunks


def chunk_document(
    segments: list[dict],
    filename: str,
    file_type: str,
    doc_meta: dict | None = None,
    min_words: int = 80,
    max_words: int = 350,
) -> list[dict]:
    """
    Divide os segmentos do documento em chunks baseados em parágrafos, com metadados completos.

    Metadados por chunk:
      source_file, file_type, chunk_id, doc_id, chunk_position, total_chunks,
      doc_title, doc_date, doc_number, page (somente PDF).
    """
    if doc_meta is None:
        doc_meta = {}

    doc_id = _doc_id(filename)
    chunks: list[dict] = []
    position = 0

    for segment in segments:
        for chunk_text in _split_paragraphs(segment['text'], min_words, max_words):
            if not chunk_text.strip():
                continue

            chunk_id = f"{filename}_chunk_{position:04d}"
            metadata: dict = {
                'source_file': filename,
                'file_type': file_type,
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'chunk_position': position,
                'doc_title': doc_meta.get('doc_title', ''),
                'doc_date': doc_meta.get('doc_date', ''),
                'doc_number': doc_meta.get('doc_number', ''),
            }
            if segment.get('page') is not None:
                metadata['page'] = segment['page']

            chunks.append({
                'text': chunk_text,
                'metadata': metadata,
                'chunk_id': chunk_id,
            })
            position += 1

    # Segunda passagem: registra total_chunks para que cada chunk conheça o tamanho do documento
    total = len(chunks)
    for chunk in chunks:
        chunk['metadata']['total_chunks'] = total

    return chunks
