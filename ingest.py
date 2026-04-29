#!/usr/bin/env python3
"""Index all documents found in data/raw/ into the local vector store."""
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from rag.loader import list_documents, load_sidecar
from rag.parser import extract_text, extract_doc_meta
from rag.chunker import chunk_document
from rag.embedder import embed_texts
from rag.vectorstore import add_chunks, count, is_indexed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Index documents from data/raw/ into the local vector store.'
    )
    parser.add_argument(
        '--reindex',
        action='store_true',
        help='Force reindex all files, even if already indexed.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    if args.reindex:
        print("\n[--reindex] Forcing full reindex of all files.\n")

    total_chunks = 0
    skipped = 0

    for doc in documents:
        if not args.reindex and is_indexed(doc['filename']):
            print(f"\nSkipping (already indexed): {doc['filename']}")
            skipped += 1
            continue

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

    if skipped:
        print(f"\nSkipped {skipped} already-indexed file(s). Use --reindex to force.")
    print(f"\nDone. Total chunks in vector store: {count()}")


if __name__ == '__main__':
    main()
