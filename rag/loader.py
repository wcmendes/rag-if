from pathlib import Path

SUPPORTED_EXTENSIONS = {'.html', '.htm', '.txt', '.pdf'}


def list_documents(raw_dir: str = 'data/raw') -> list[dict]:
    """Return sorted list of supported documents found in raw_dir."""
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
