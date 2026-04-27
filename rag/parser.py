from __future__ import annotations

import re
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

_MONTHS = (
    'janeiro|fevereiro|março|abril|maio|junho|'
    'julho|agosto|setembro|outubro|novembro|dezembro'
)
_DATE_LONG = re.compile(
    rf'\d{{1,2}}\s+de\s+(?:{_MONTHS})\s+de\s+20\d{{2}}',
    re.IGNORECASE,
)
_DATE_SHORT = re.compile(r'\d{2}/\d{2}/20\d{2}')
_DOC_NUMBER = re.compile(
    r'(?:Resolução|Portaria|Instrução Normativa|Decreto|Ordem de Serviço)'
    r'.{0,40}n[ºo°]?\s*\.?\s*[\d]+(?:[/\-]\d+)?',
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def parse_html(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
        tag.decompose()
    return clean_text(soup.get_text(separator='\n'))


def parse_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return clean_text(f.read())


def parse_pdf(path: str) -> list[dict]:
    """Return one segment per page: {'page': int, 'text': str}."""
    doc = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = clean_text(page.get_text())
        if text:
            pages.append({'page': page_num, 'text': text})
    doc.close()
    return pages


def extract_text(path: str, file_type: str) -> list[dict]:
    """
    Returns list of {'text': str, 'page': int|None}.
    PDF: one entry per page. HTML/TXT: single entry.
    """
    if file_type == 'pdf':
        return parse_pdf(path)
    if file_type == 'html':
        return [{'text': parse_html(path), 'page': None}]
    return [{'text': parse_txt(path), 'page': None}]


def extract_doc_meta(segments: list[dict], path: str = '', file_type: str = '') -> dict:
    """
    Extract document-level metadata from parsed content.
    Returns: doc_title, doc_date, doc_number.
    """
    full_text = '\n\n'.join(s['text'] for s in segments)
    meta: dict = {'doc_title': '', 'doc_date': '', 'doc_number': ''}

    # --- doc_title ---
    # For HTML, prefer the <title> or first <h1>/<h2> tag
    if file_type == 'html' and path:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            for tag_name in ('title', 'h1', 'h2'):
                tag = soup.find(tag_name)
                if tag and tag.get_text(strip=True):
                    meta['doc_title'] = clean_text(tag.get_text())[:200]
                    break
        except Exception:
            pass

    # Fallback: first meaningful line of text
    if not meta['doc_title']:
        for line in full_text.split('\n'):
            line = line.strip()
            if len(line) > 10:
                meta['doc_title'] = line[:200]
                break

    # --- doc_number ---
    match = _DOC_NUMBER.search(full_text)
    if match:
        meta['doc_number'] = match.group(0).strip()
        # Use as title if it's more descriptive
        if not meta['doc_title'] or len(meta['doc_number']) > len(meta['doc_title']):
            meta['doc_title'] = meta['doc_number']

    # --- doc_date ---
    match = _DATE_LONG.search(full_text)
    if match:
        meta['doc_date'] = match.group(0)
    else:
        match = _DATE_SHORT.search(full_text)
        if match:
            meta['doc_date'] = match.group(0)

    return meta
