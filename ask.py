#!/usr/bin/env python3
"""Query the indexed documents and print the answer with sources."""
import sys
from dotenv import load_dotenv

load_dotenv()

from rag.retriever import retrieve
from rag.generator import generate_answer


def query_rag(question: str, n_results: int = 5) -> dict:
    """
    Structured RAG result for programmatic / evaluation use.

    Returns:
      question    - the input question
      answer      - generated answer (or a 'not found' message)
      contexts    - list of chunk texts used as context
      source_ids  - chunk_id for each context chunk
      file_names  - source_file for each context chunk
    """
    chunks = retrieve(question, n_results=n_results)
    if not chunks:
        return {
            "question": question,
            "answer": "Nenhum documento foi encontrado no índice. Execute python ingest.py primeiro.",
            "contexts": [],
            "source_ids": [],
            "file_names": [],
        }
    answer = generate_answer(question, chunks)
    return {
        "question": question,
        "answer": answer,
        "contexts": [c["text"] for c in chunks],
        "source_ids": [c["metadata"].get("chunk_id", "") for c in chunks],
        "file_names": [c["metadata"].get("source_file", "") for c in chunks],
    }


def _format_source(meta: dict) -> str:
    source = meta.get('source_file', 'unknown')
    chunk_id = meta.get('chunk_id', '')
    page = meta.get('page')
    base = f"{source} | {chunk_id}"
    return f"{base} | página {page}" if page else base


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ask.py \"sua pergunta aqui\"")
        sys.exit(1)

    question = ' '.join(sys.argv[1:])

    print(f"\nPergunta:\n{question}\n")

    chunks = retrieve(question, n_results=5)

    if not chunks:
        print("Resposta:")
        print("Nenhum documento foi encontrado no índice. Execute python ingest.py primeiro.\n")
        sys.exit(0)

    print("Gerando resposta...\n")
    answer = generate_answer(question, chunks)

    print(f"Resposta:\n{answer}\n")

    print("Fontes:")
    seen: set[str] = set()
    for chunk in chunks:
        label = _format_source(chunk['metadata'])
        if label not in seen:
            print(f"  - {label}")
            seen.add(label)
    print()


if __name__ == '__main__':
    main()
