import os
import requests


def _format_context(chunks: list[dict]) -> str:
    """Format chunks with their document metadata as visible headers."""
    parts = []
    for chunk in chunks:
        meta = chunk['metadata']

        header_parts = [f"Arquivo: {meta.get('source_file', '?')}"]
        if meta.get('doc_number'):
            header_parts.append(f"Documento: {meta['doc_number']}")
        elif meta.get('doc_title'):
            header_parts.append(f"Título: {meta['doc_title']}")
        if meta.get('doc_date'):
            header_parts.append(f"Data de publicação: {meta['doc_date']}")
        if meta.get('page'):
            header_parts.append(f"Página: {meta['page']}")

        header = ' | '.join(header_parts)
        parts.append(f"[{header}]\n{chunk['text']}")

    return '\n\n---\n\n'.join(parts)


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    return (
        "INSTRUÇÃO OBRIGATÓRIA: Você DEVE responder EXCLUSIVAMENTE em português do Brasil. "
        "Não use nenhuma outra língua, independentemente do idioma dos documentos ou do modelo.\n\n"
        "Você é um assistente especializado em documentos normativos institucionais.\n"
        "Responda à pergunta abaixo com base APENAS nos trechos de documentos fornecidos.\n\n"
        "REGRA PARA CONFLITOS: Se documentos diferentes apresentarem informações contraditórias "
        "sobre o mesmo assunto, identifique explicitamente o conflito. Cite qual norma é mais "
        "antiga e qual é mais recente, e o que cada uma estabelece. "
        "Exemplo: 'A Portaria X (publicada em 2022) estabelecia Y. "
        "A Resolução Z (publicada em 2024) alterou essa regra para W.'\n\n"
        "Se as informações nos documentos não forem suficientes para responder, "
        "diga claramente que não encontrou evidências nos documentos indexados.\n\n"
        f"Documentos:\n{context}\n\n"
        f"Pergunta: {question}\n\n"
        "Resposta em português do Brasil:"
    )


def _call_ollama(prompt: str) -> str:
    model = os.getenv('OLLAMA_MODEL', 'llama3.2')
    url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    response = requests.post(
        f'{url}/api/generate',
        json={'model': model, 'prompt': prompt, 'stream': False},
        timeout=600,
    )
    if not response.ok:
        print(f"Ollama error {response.status_code}: {response.text}")
    response.raise_for_status()
    return response.json()['response'].strip()


def _call_openai(prompt: str) -> str:
    import openai
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    response = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Não foram encontrados documentos relevantes para responder à pergunta."

    prompt = _build_prompt(question, chunks)
    provider = os.getenv('LLM_PROVIDER', 'ollama').lower()

    if provider == 'openai':
        return _call_openai(prompt)
    return _call_ollama(prompt)
