import os
import requests

SYSTEM_PROMPT = (
    "Você é um assistente especializado em documentos normativos institucionais.\n\n"
    "REGRAS OBRIGATÓRIAS — siga-as sem exceção:\n"
    "1. Responda SOMENTE com base nos trechos de documentos fornecidos na mensagem do usuário.\n"
    "2. NÃO use conhecimento externo, treinamento prévio ou inferências além do que está escrito nos documentos.\n"
    "3. NÃO complete, extrapole ou suponha informações ausentes nos documentos.\n"
    "4. Se os documentos não contiverem informação suficiente, responda exatamente: "
    "'Não encontrei evidências suficientes nos documentos indexados para responder a esta pergunta.'\n"
    "5. Cite apenas fatos que estejam explicitamente no texto dos documentos fornecidos.\n"
    "6. Se documentos diferentes apresentarem informações contraditórias, descreva o que cada um diz "
    "citando apenas o que está escrito — não interprete nem julgue qual prevalece.\n"
    "7. Responda sempre em português do Brasil."
)


def _format_context(chunks: list[dict]) -> str:
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


def _build_user_message(question: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    return f"Documentos:\n{context}\n\nPergunta: {question}"


def _call_ollama(question: str, chunks: list[dict]) -> str:
    model = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')
    url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': _build_user_message(question, chunks)},
    ]
    response = requests.post(
        f'{url}/api/chat',
        json={'model': model, 'messages': messages, 'stream': False},
        timeout=600,
    )
    if not response.ok:
        print(f"Ollama error {response.status_code}: {response.text}")
    response.raise_for_status()
    return response.json()['message']['content'].strip()


def _call_openai(question: str, chunks: list[dict]) -> str:
    import openai
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': _build_user_message(question, chunks)},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Não foram encontrados documentos relevantes para responder à pergunta."

    provider = os.getenv('LLM_PROVIDER', 'ollama').lower()

    if provider == 'openai':
        return _call_openai(question, chunks)
    return _call_ollama(question, chunks)
