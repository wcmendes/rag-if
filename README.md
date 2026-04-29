# rag-if

Sistema RAG local em Python para consulta de documentos normativos institucionais por linha de comando.

## Requisitos

- Python 3.11+
- [Ollama](https://ollama.com) instalado e rodando localmente (padrão), **ou** chave da OpenAI

## Instalação

```bash
# 1. Clone e entre no diretório
git clone <repo-url>
cd rag-if

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure o ambiente
cp .env.example .env
# Edite .env se quiser trocar de modelo ou provider
```

## Configuração do LLM

### Opção A — Ollama (local, padrão)

```bash
# Instale o Ollama: https://ollama.com
ollama pull llama3.2
```

Deixe `LLM_PROVIDER=ollama` no `.env`.

### Opção B — OpenAI

Edite `.env`:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Instale o pacote opcional:

```bash
pip install openai
```

## Uso

### 1. Indexar documentos

Coloque arquivos `.html`, `.txt` ou `.pdf` em `data/raw/` e execute:

```bash
python ingest.py
```

Saída esperada:

```
Found 2 document(s):
  - resolucao_001_2024.html (html)
  - resolucao_002_2024.pdf (pdf)

Processing: resolucao_001_2024.html
  Segments extracted: 1
  Chunks created:     12
  [embedder] Loading model: paraphrase-multilingual-MiniLM-L12-v2
  Indexed: 12 chunk(s)
...
Done. Total chunks in vector store: 25
```

O modelo de embeddings (~450 MB) é baixado automaticamente na primeira execução.

Para **reindexar** após adicionar novos arquivos, basta rodar `python ingest.py` novamente — os chunks existentes são atualizados via upsert.

### 2. Consultar

```bash
python ask.py "Quais documentos tratam de afastamento para capacitação?"
```

Saída esperada:

```
Pergunta:
Quais documentos tratam de afastamento para capacitação?

Gerando resposta...

Resposta:
Com base nos documentos indexados, a Resolução 001/2024 trata de afastamento
para capacitação nos artigos 5º e 12º, estabelecendo os requisitos e prazos...

Fontes:
  - resolucao_001_2024.html | resolucao_001_2024.html_chunk_0003
  - resolucao_002_2024.pdf  | resolucao_002_2024.pdf_chunk_0007 | página 3
```

## Estrutura do projeto

```
rag-if/
├── ingest.py          # indexação
├── ask.py             # consulta
├── requirements.txt
├── .env.example
├── rag/
│   ├── loader.py      # lista arquivos de data/raw/
│   ├── parser.py      # extrai texto (HTML, TXT, PDF)
│   ├── chunker.py     # divide em chunks com overlap
│   ├── embedder.py    # sentence-transformers
│   ├── vectorstore.py # ChromaDB persistente
│   ├── retriever.py   # busca semântica
│   └── generator.py   # chamada ao LLM
├── data/
│   └── raw/           # coloque seus documentos aqui
└── vectordb/          # banco vetorial local (gerado automaticamente)
```

## Tipos de arquivo suportados

| Extensão | Parser |
|---|---|
| `.html`, `.htm` | BeautifulSoup4 |
| `.txt` | leitura direta |
| `.pdf` | PyMuPDF (texto nativo; sem OCR) |

> PDFs escaneados (imagem) não são suportados nesta versão.

## Variáveis de ambiente

Veja [.env.example](.env.example) para todas as opções disponíveis.

---

## Pipeline de avaliação RAG com RAGAS

O script [orchestrator_rag_eval.py](orchestrator_rag_eval.py) automatiza a avaliação do sistema em três etapas:

```
perguntas.json  →  (collect)  →  respostas_rag.json  →  (evaluate)  →  resultados_ragas/
```

### Quando o RAGAS entra no pipeline

O RAGAS **não é usado durante a geração das respostas**. Ele entra **somente na etapa de avaliação**, depois que todas as respostas já foram coletadas. Isso significa que você pode rodar `collect` com o Ollama local, e depois rodar `evaluate` com uma chave OpenAI (que o RAGAS usa como juiz).

### 1. Instalar dependências do RAGAS

```bash
pip install ragas langchain-openai openai
```

> Se quiser usar Ollama como juiz (mais lento, menor qualidade de avaliação):
> ```bash
> pip install langchain-community
> ```

### 2. Configurar o modelo julgador

Edite seu `.env` e adicione:

```env
# Juiz do RAGAS — use um modelo superior ao seu RAG principal
RAGAS_LLM_PROVIDER=openai
RAGAS_JUDGE_MODEL=gpt-4o
RAGAS_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
```

> Para usar Ollama como juiz:
> ```env
> RAGAS_LLM_PROVIDER=ollama
> RAGAS_JUDGE_MODEL=llama3.3
> RAGAS_EMBEDDING_MODEL=nomic-embed-text
> ```

### 3. Preparar o arquivo de perguntas

Crie `perguntas.json` com o seguinte formato (veja [perguntas_exemplo.json](perguntas_exemplo.json)):

```json
[
  {
    "id": 1,
    "question": "Sua pergunta aqui",
    "ground_truth": "Resposta correta esperada"
  }
]
```

### 4. Coletar respostas do RAG

Certifique-se de que o `ingest.py` já foi executado e o vectordb está populado. Depois:

```bash
python orchestrator_rag_eval.py --mode collect --input perguntas.json --output respostas_rag.json
```

Isso gera `respostas_rag.json` (e também `.jsonl` e `.csv`) com:

```json
[
  {
    "id": 1,
    "question": "...",
    "ground_truth": "...",
    "answer": "Resposta gerada pelo RAG",
    "contexts": ["chunk 1", "chunk 2", "..."]
  }
]
```

> **Checkpoint automático**: se a coleta for interrompida, reexecute o mesmo comando.
> Perguntas já processadas são automaticamente puladas.

### 5. Avaliar com RAGAS

```bash
python orchestrator_rag_eval.py --mode evaluate --input respostas_rag.json --output-dir ./resultados_ragas
```

Métricas calculadas (quando disponíveis na versão instalada):

| Métrica | O que avalia |
|---|---|
| `faithfulness` | A resposta é fiel aos documentos recuperados? |
| `answer_relevancy` | A resposta é relevante para a pergunta? |
| `context_precision` | Os chunks mais relevantes estão no topo? |
| `context_recall` | O contexto cobre o que a resposta de referência exige? |
| `answer_correctness` | A resposta está correta comparada ao ground truth? |

Os resultados são salvos em `resultados_ragas/ragas_results_<timestamp>.csv` e `.json`.

### 6. Rodar tudo de uma vez

```bash
python orchestrator_rag_eval.py --mode all \
  --input perguntas.json \
  --output respostas_rag.json \
  --output-dir ./resultados_ragas
```

### Opções da CLI

```
--mode          collect | evaluate | all
--input         arquivo de entrada (padrão: perguntas.json)
--output        arquivo de saída enriquecido (padrão: respostas_rag.json)
--output-dir    pasta para resultados do RAGAS (padrão: ./resultados_ragas)
--n-results     chunks recuperados por pergunta (padrão: 5)
```
