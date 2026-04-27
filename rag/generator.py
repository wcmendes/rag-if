import os
import requests


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context = '\n\n---\n\n'.join(c['text'] for c in chunks)
    return (
        "INSTRUÇÃO OBRIGATÓRIA: Você DEVE responder EXCLUSIVAMENTE em português do Brasil. "
        "Não use nenhuma outra língua, independentemente do idioma dos documentos ou do modelo.\n\n"
        "Você é um assistente especializado em documentos normativos institucionais.\n"
        "Responda à pergunta abaixo com base APENAS nos trechos de documentos fornecidos.\n"
        "Se as informações nos documentos não forem suficientes, diga claramente que não encontrou evidências.\n\n"
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
        timeout=300,
    )
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
