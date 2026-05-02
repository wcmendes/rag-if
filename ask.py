#!/usr/bin/env python3
"""Consulta os documentos indexados e imprime a resposta com as fontes."""

__author__ = "William Mendes"

import sys
from dotenv import load_dotenv

load_dotenv()

from rag.retriever import retrieve
from rag.generator import generate_answer


def query_rag(question: str, n_results: int = 5) -> dict:
    """
    Resultado estruturado do RAG para uso programático / avaliação.

    Retorna:
      question    - a pergunta de entrada
      answer      - resposta gerada (ou mensagem de 'não encontrado')
      contexts    - lista de textos dos chunks usados como contexto
      source_ids  - chunk_id de cada chunk de contexto
      file_names  - source_file de cada chunk de contexto
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
