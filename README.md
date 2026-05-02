<div align="center">

# rag-if

Sistema RAG local para consulta de documentos normativos de Institutos Federais brasileiros  
*Local RAG system for querying institutional normative documents of Brazilian Federal Institutions*

**William Mendes**  
Diretor de Gestão de Tecnologia da Informação · [IFMA](https://ifma.edu.br)  
Mestre em Engenharia Elétrica — UFMA · Doutorando em Administração · [FUCAPE Business School](https://fucape.br)

[![Lattes](https://img.shields.io/badge/Currículo-Lattes%20CNPq-1f6feb?style=flat-square)](https://lattes.cnpq.br/7726054867638395)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Embedding](https://img.shields.io/badge/Embedding-BAAI%2Fbge--m3-orange?style=flat-square)](https://huggingface.co/BAAI/bge-m3)

---

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

</div>

---

<a id="-português"></a>
## 🇧🇷 Português

Aplicação de linha de comando em Python que indexa documentos HTML, PDF e texto simples em um vetor store local e recupera respostas fundamentadas em contexto via LLM — sem envio de dados para fora da infraestrutura quando operando com Ollama.

### Requisitos

- Python 3.11+
- [Ollama](https://ollama.com) instalado e em execução local (padrão), **ou** chave de API da OpenAI

### Instalação

```bash
# 1. Clone e entre no diretório
git clone <repo-url>
cd rag-if

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure o ambiente
cp .env.example .env
# Edite .env para trocar modelo ou provider
```

### Configuração do LLM

**Opção A — Ollama (local, padrão)**

```bash
ollama pull qwen2.5:7b
```

Mantenha `LLM_PROVIDER=ollama` no `.env`.

**Opção B — OpenAI**

Edite `.env`:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

```bash
pip install openai
```

### Uso

**1. Indexar documentos**

Coloque arquivos `.html`, `.txt` ou `.pdf` em `data/raw/` e execute:

> **Nota:** Os documentos normativos institucionais utilizados nos experimentos (corpus documental) não estão incluídos neste repositório.

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
  [embedder] Loading model: BAAI/bge-m3
  Indexed: 12 chunk(s)
...
Done. Total chunks in vector store: 25
```

O modelo de embeddings (~1,1 GB) é baixado automaticamente na primeira execução.
Para reindexar após adicionar arquivos, basta rodar `python ingest.py` novamente. Use `--reindex` para forçar reindexação completa.

**2. Consultar**

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

### Estrutura do projeto

```
rag-if/
├── ingest.py                # indexação de documentos
├── ask.py                   # interface de consulta
├── inspect_chunks.py        # depuração: lista chunks indexados
├── orchestrator_rag_eval.py # pipeline de avaliação automatizada
├── requirements.txt
├── .env.example
├── rag/
│   ├── loader.py            # descobre arquivos em data/raw/
│   ├── parser.py            # extrai texto (HTML, TXT, PDF)
│   ├── chunker.py           # divisão em chunks por parágrafo
│   ├── embedder.py          # BAAI/bge-m3 via sentence-transformers
│   ├── vectorstore.py       # ChromaDB persistente
│   ├── retriever.py         # busca semântica + expansão de contexto
│   └── generator.py         # chamada ao LLM (Ollama ou OpenAI)
├── data/
│   └── raw/                 # coloque seus documentos aqui
└── vectordb/                # banco vetorial local (gerado automaticamente)
```

### Tipos de arquivo suportados

| Extensão | Parser |
|---|---|
| `.html`, `.htm` | BeautifulSoup4 |
| `.txt` | leitura direta |
| `.pdf` | PyMuPDF (texto nativo; sem OCR) |

> PDFs escaneados (somente imagem) não são suportados nesta versão.

### Variáveis de ambiente

Veja [.env.example](.env.example) para todas as opções disponíveis.

---

### Pipeline de avaliação RAG com RAGAS

O script [orchestrator_rag_eval.py](orchestrator_rag_eval.py) automatiza a avaliação em três etapas:

```
perguntas.json  →  (collect)  →  respostas_rag.json  →  (evaluate)  →  resultados_ragas/
```

O RAGAS **não é usado durante a geração das respostas** — entra somente na etapa de avaliação, depois que todas as respostas já foram coletadas. Você pode rodar `collect` com Ollama local e depois `evaluate` com uma chave OpenAI.

**Instalar dependências do RAGAS**

```bash
pip install ragas langchain-openai openai
# Para usar Ollama como juiz: pip install langchain-ollama
```

**Configurar o modelo julgador** (adicione ao `.env`):

```env
RAGAS_LLM_PROVIDER=openai
RAGAS_JUDGE_MODEL=gpt-4o-mini
RAGAS_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
```

**Preparar arquivo de perguntas** (`perguntas.json`):

```json
[
  { "id": 1, "question": "Sua pergunta aqui", "ground_truth": "Resposta correta esperada" }
]
```

**Coletar respostas do RAG:**

```bash
python orchestrator_rag_eval.py --mode collect --input perguntas.json --output respostas_rag.json
```

> Checkpoint automático: se interrompido, reexecute o mesmo comando — perguntas já processadas são puladas.

**Avaliar com RAGAS:**

```bash
python orchestrator_rag_eval.py --mode evaluate --input respostas_rag.json --output-dir ./resultados_ragas
```

| Métrica | O que avalia |
|---|---|
| `faithfulness` | A resposta é fiel aos documentos recuperados? |
| `answer_relevancy` | A resposta é relevante para a pergunta? |
| `context_precision` | Os chunks mais relevantes estão no topo? |
| `context_recall` | O contexto cobre o que a resposta de referência exige? |
| `answer_correctness` | A resposta está correta comparada ao ground truth? |

Resultados salvos em `resultados_ragas/ragas_results_<timestamp>.csv` e `.json`.

**Rodar tudo de uma vez:**

```bash
python orchestrator_rag_eval.py --mode all \
  --input perguntas.json \
  --output respostas_rag.json \
  --output-dir ./resultados_ragas
```

**Opções da CLI:**

```
--mode          collect | evaluate | all
--input         arquivo de entrada        (padrão: perguntas.json)
--output        dataset de saída          (padrão: respostas_rag.json)
--output-dir    pasta para resultados     (padrão: ./resultados_ragas)
--n-results     chunks por pergunta       (padrão: 5)
```

---

<a id="-english"></a>
## 🇺🇸 English

A command-line Python application that indexes HTML, PDF, and plain-text documents into a local vector store and retrieves context-grounded answers using a Large Language Model — with no data sent outside your infrastructure when running in Ollama mode.

### Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally (default), **or** an OpenAI API key

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd rag-if

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env to change the model or provider
```

### LLM Configuration

**Option A — Ollama (local, default)**

```bash
ollama pull qwen2.5:7b
```

Set `LLM_PROVIDER=ollama` in `.env`.

**Option B — OpenAI**

Edit `.env`:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

```bash
pip install openai
```

### Usage

**1. Index documents**

Place `.html`, `.txt`, or `.pdf` files in `data/raw/` and run:

> **Note:** The institutional normative documents used in the experiments (document corpus) are not included in this repository.

```bash
python ingest.py
```

Expected output:

```
Found 2 document(s):
  - resolucao_001_2024.html (html)
  - resolucao_002_2024.pdf (pdf)

Processing: resolucao_001_2024.html
  Segments extracted: 1
  Chunks created:     12
  [embedder] Loading model: BAAI/bge-m3
  Indexed: 12 chunk(s)
...
Done. Total chunks in vector store: 25
```

The embedding model (~1.1 GB) is downloaded automatically on the first run.
To reindex after adding files, simply run `python ingest.py` again. Use `--reindex` to force a full reindex.

**2. Query**

```bash
python ask.py "Quais documentos tratam de afastamento para capacitação?"
```

Expected output:

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

### Project Structure

```
rag-if/
├── ingest.py                # document indexing
├── ask.py                   # query interface
├── inspect_chunks.py        # debug: list all indexed chunks
├── orchestrator_rag_eval.py # automated evaluation pipeline
├── requirements.txt
├── .env.example
├── rag/
│   ├── loader.py            # discover files in data/raw/
│   ├── parser.py            # extract text (HTML, TXT, PDF)
│   ├── chunker.py           # paragraph-based chunking
│   ├── embedder.py          # BAAI/bge-m3 via sentence-transformers
│   ├── vectorstore.py       # persistent ChromaDB
│   ├── retriever.py         # semantic search + context window expansion
│   └── generator.py         # LLM call (Ollama or OpenAI)
├── data/
│   └── raw/                 # place your documents here
└── vectordb/                # local vector store (auto-generated)
```

### Supported File Types

| Extension | Parser |
|---|---|
| `.html`, `.htm` | BeautifulSoup4 |
| `.txt` | direct read |
| `.pdf` | PyMuPDF (native text; no OCR) |

> Scanned PDFs (image-only) are not supported in this version.

### Environment Variables

See [.env.example](.env.example) for all available options.

---

### RAG Evaluation Pipeline (RAGAS)

The script [orchestrator_rag_eval.py](orchestrator_rag_eval.py) automates evaluation in three stages:

```
perguntas.json  →  (collect)  →  respostas_rag.json  →  (evaluate)  →  resultados_ragas/
```

RAGAS is **not used during answer generation** — it runs only in the evaluation stage, after all answers have been collected. You can run `collect` with local Ollama and then `evaluate` with an OpenAI key.

**Install RAGAS dependencies:**

```bash
pip install ragas langchain-openai openai
# To use Ollama as judge: pip install langchain-ollama
```

**Configure the judge model** (add to `.env`):

```env
RAGAS_LLM_PROVIDER=openai
RAGAS_JUDGE_MODEL=gpt-4o-mini
RAGAS_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
```

**Prepare the question file** (`perguntas.json`):

```json
[
  { "id": 1, "question": "Your question here", "ground_truth": "Expected correct answer" }
]
```

**Collect RAG answers:**

```bash
python orchestrator_rag_eval.py --mode collect --input perguntas.json --output respostas_rag.json
```

> Automatic checkpoint: if interrupted, re-run the same command — already-processed questions are skipped.

**Evaluate with RAGAS:**

```bash
python orchestrator_rag_eval.py --mode evaluate --input respostas_rag.json --output-dir ./resultados_ragas
```

| Metric | What it measures |
|---|---|
| `faithfulness` | Is the answer faithful to the retrieved documents? |
| `answer_relevancy` | Is the answer relevant to the question? |
| `context_precision` | Are the most relevant chunks ranked highest? |
| `context_recall` | Does the retrieved context cover what the reference answer requires? |
| `answer_correctness` | Is the answer correct compared to the ground truth? |

Results are saved to `resultados_ragas/ragas_results_<timestamp>.csv` and `.json`.

**Run everything at once:**

```bash
python orchestrator_rag_eval.py --mode all \
  --input perguntas.json \
  --output respostas_rag.json \
  --output-dir ./resultados_ragas
```

**CLI Options:**

```
--mode          collect | evaluate | all
--input         input file             (default: perguntas.json)
--output        enriched output file   (default: respostas_rag.json)
--output-dir    directory for results  (default: ./resultados_ragas)
--n-results     chunks per question    (default: 5)
```
