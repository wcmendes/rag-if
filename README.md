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
