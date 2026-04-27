#!/usr/bin/env python3
"""List all indexed chunks. Useful for debugging."""
import sys
from dotenv import load_dotenv

load_dotenv()

from rag.vectorstore import get_collection


def main() -> None:
    collection = get_collection()
    total = collection.count()
    print(f"Total de chunks indexados: {total}\n")

    if total == 0:
        print("Nenhum chunk encontrado. Execute python ingest.py primeiro.")
        return

    # Optional filter by filename: python inspect_chunks.py arquivo1.html
    source_filter = sys.argv[1] if len(sys.argv) > 1 else None

    results = collection.get(include=['documents', 'metadatas'])

    for doc, meta in zip(results['documents'], results['metadatas']):
        if source_filter and source_filter not in meta.get('source_file', ''):
            continue
        page = f" | página {meta['page']}" if 'page' in meta else ''
        print(f"[{meta['chunk_id']}{page}]")
        print(doc[:300].replace('\n', ' '))
        print()


if __name__ == '__main__':
    main()
