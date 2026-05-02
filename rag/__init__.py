"""
Pacote rag-if — pipeline RAG local para consulta de documentos normativos institucionais.

Módulos:
  loader      — descobre arquivos suportados em data/raw/
  parser      — extrai texto de HTML, TXT e PDF
  chunker     — divide documentos em chunks baseados em parágrafos
  embedder    — gera embeddings com BAAI/bge-m3 (sentence-transformers)
  vectorstore — persistência e busca no ChromaDB
  retriever   — busca semântica com expansão de janela de contexto
  generator   — geração de resposta via LLM (Ollama ou OpenAI)
"""

__author__ = "William Mendes"
__institution__ = "Instituto Federal do Maranhão (IFMA) · FUCAPE Business School"
__lattes__ = "https://lattes.cnpq.br/7726054867638395"
