#!/usr/bin/env python3
"""Index all documents found in data/raw/ into the local vector store."""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from rag.loader import list_documents, load_sidecar
from rag.parser import extract_text, extract_doc_meta
from rag.chunker import chunk_document
from rag.embedder import embed_texts
from rag.vectorstore import add_chunks, count


def main() -> None:
    raw_dir = 'data/raw'

    try:
        documents = list_documents(raw_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not documents:
        print(f"No supported documents found in {raw_dir}/")
        print("Supported types: .html, .htm, .txt, .pdf")
        sys.exit(0)

    sidecar = load_sidecar(Path(raw_dir))

    print(f"Found {len(documents)} document(s):")
    for doc in documents:
        print(f"  - {doc['filename']} ({doc['file_type']})")

    total_chunks = 0

    for doc in documents:
        print(f"\nProcessing: {doc['filename']}")

        segments = extract_text(doc['path'], doc['file_type'])
        print(f"  Segments extracted: {len(segments)}")

        # Auto-extract metadata from content, then apply manual overrides
        doc_meta = extract_doc_meta(segments, doc['path'], doc['file_type'])
        manual = sidecar.get(doc['filename'], {})
        if manual.get('title'):
            doc_meta['doc_title'] = manual['title']
        if manual.get('date'):
            doc_meta['doc_date'] = manual['date']
        if manual.get('number'):
            doc_meta['doc_number'] = manual['number']

        print(f"  Title : {doc_meta['doc_title'] or '(not detected)'}")
        print(f"  Date  : {doc_meta['doc_date'] or '(not detected)'}")
        print(f"  Number: {doc_meta['doc_number'] or '(not detected)'}")

        chunks = chunk_document(segments, doc['filename'], doc['file_type'], doc_meta)
        print(f"  Chunks created: {len(chunks)}")

        if not chunks:
            print("  Skipping (no content after chunking)")
            continue

        embeddings = embed_texts([c['text'] for c in chunks])
        add_chunks(chunks, embeddings)
        total_chunks += len(chunks)
        print(f"  Indexed: {len(chunks)} chunk(s)")

    print(f"\nDone. Total chunks in vector store: {count()}")


if __name__ == '__main__':
    main()
