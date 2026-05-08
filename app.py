#!/usr/bin/env python3
"""Web interface for the RAG-IF document assistant."""
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, send_from_directory
from rag.retriever import retrieve
from rag.generator import generate_answer

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory('images', filename)


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = (data or {}).get('question', '').strip()

        if not question:
            return jsonify({'error': 'Pergunta não informada.'}), 400

        chunks = retrieve(question, n_results=5)

        if not chunks:
            return jsonify({
                'answer': 'Nenhum documento foi encontrado no índice. Execute python ingest.py primeiro.',
                'sources': [],
            })

        answer = generate_answer(question, chunks)

        seen: set[str] = set()
        sources = []
        for chunk in chunks:
            meta = chunk['metadata']
            chunk_id = meta.get('chunk_id', '')
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            sources.append({
                'source_file':    meta.get('source_file', ''),
                'doc_number':     meta.get('doc_number', ''),
                'doc_title':      meta.get('doc_title', ''),
                'doc_date':       meta.get('doc_date', ''),
                'chunk_id':       chunk_id,
                'chunk_position': meta.get('chunk_position', ''),
                'total_chunks':   meta.get('total_chunks', ''),
                'page':           meta.get('page', ''),
            })

        return jsonify({'answer': answer, 'sources': sources})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
