from __future__ import annotations

import hashlib
import re


def _doc_id(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:8]


def _split_paragraphs(text: str, min_words: int = 80, max_words: int = 350) -> list[str]:
    """
    Split text at blank lines, then group small paragraphs together
    and break overly long ones, preserving paragraph boundaries.
    """
    raw = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for para in raw:
        word_count = len(para.split())

        if word_count > max_words:
            # Flush accumulated parts first
            if current_parts:
                chunks.append('\n\n'.join(current_parts))
                current_parts = []
                current_words = 0
            # Break the long paragraph by word count
            words = para.split()
            for i in range(0, len(words), max_words):
                chunks.append(' '.join(words[i:i + max_words]))

        elif current_words + word_count > max_words and current_words >= min_words:
            # Current group is full — flush and start new
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
    Split document segments into paragraph-based chunks with full metadata.

    Metadata per chunk:
      source_file, file_type, chunk_id, doc_id, chunk_position, total_chunks,
      doc_title, doc_date, doc_number, page (PDF only).
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

    # Second pass: stamp total_chunks so every chunk knows the document size
    total = len(chunks)
    for chunk in chunks:
        chunk['metadata']['total_chunks'] = total

    return chunks
