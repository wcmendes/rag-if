#!/usr/bin/env python3
"""Index all documents found in data/raw/ into the local vector store."""
import sys
from dotenv import load_dotenv

load_dotenv()

from rag.loader import list_documents
from rag.parser import extract_text
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

    print(f"Found {len(documents)} document(s):")
    for doc in documents:
        print(f"  - {doc['filename']} ({doc['file_type']})")

    total_chunks = 0

    for doc in documents:
        print(f"\nProcessing: {doc['filename']}")

        segments = extract_text(doc['path'], doc['file_type'])
        print(f"  Segments extracted: {len(segments)}")

        chunks = chunk_document(segments, doc['filename'], doc['file_type'])
        print(f"  Chunks created:     {len(chunks)}")

        if not chunks:
            print("  Skipping (no content after chunking)")
            continue

        embeddings = embed_texts([c['text'] for c in chunks])
        add_chunks(chunks, embeddings)
        total_chunks += len(chunks)
        print(f"  Indexed:            {len(chunks)} chunk(s)")

    print(f"\nDone. Total chunks in vector store: {count()}")


if __name__ == '__main__':
    main()
