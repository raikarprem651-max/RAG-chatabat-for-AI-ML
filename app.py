"""
app.py
------
Flask web app for the RAG chatbot.

Routes:
    GET  /            -> chat UI
    POST /api/chat     -> {"message": "..."} -> {"answer": "...", "sources": [...]}
    POST /api/ingest    -> re-index documents in data/sample_docs (or a posted folder)
"""

from flask import Flask, request, jsonify, render_template
from rag_engine import RAGEngine

app = Flask(__name__)
engine = RAGEngine()


@app.route("/")
def index():
    has_index = engine.index is not None and engine.index.ntotal > 0
    return render_template("index.html", has_index=has_index)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query = (data.get("message") or "").strip()
    if not query:
        return jsonify({"error": "Empty message"}), 400

    result = engine.answer(query)
    return jsonify(result)


@app.route("/api/ingest", methods=["POST"])
def ingest():
    data = request.get_json(silent=True) or {}
    folder = data.get("folder", "data/sample_docs")
    try:
        n_chunks = engine.build_index_from_folder(folder)
        return jsonify({"status": "ok", "chunks_indexed": n_chunks})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
