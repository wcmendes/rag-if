"""Descobre e lista os documentos suportados em data/raw/ para indexação."""

__author__ = "William Mendes"

import json
from pathlib import Path

SUPPORTED_EXTENSIONS = {'.html', '.htm', '.txt', '.pdf'}


def load_sidecar(raw_dir: Path) -> dict:
    """
    Carrega o metadata.json opcional de raw_dir.
    Formato: { "filename.html": { "title": "...", "date": "...", "number": "..." } }
    Os valores aqui sobrescrevem o que é extraído automaticamente do conteúdo.
    """
    sidecar = raw_dir / 'metadata.json'
    if sidecar.exists():
        with open(sidecar, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def list_documents(raw_dir: str = 'data/raw') -> list[dict]:
    """Retorna lista ordenada de documentos suportados encontrados em raw_dir."""
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Directory not found: {raw_dir}")

    documents = []
    for file_path in sorted(raw_path.iterdir()):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        if ext in ('.html', '.htm'):
            file_type = 'html'
        else:
            file_type = ext.lstrip('.')

        documents.append({
            'path': str(file_path),
            'filename': file_path.name,
            'file_type': file_type,
        })

    return documents
