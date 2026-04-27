import re
import fitz  # PyMuPDF
from bs4 import BeautifulSoup


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
