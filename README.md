# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot: it answers questions grounded
in your own documents by combining semantic search (embeddings + FAISS) with
an LLM (Claude) for generation.

## How it works

```
 Documents (.txt/.pdf/.docx)
        |
        v
   Chunking (500 chars, 80 overlap)
        |
        v
   Embedding (all-MiniLM-L6-v2, sentence-transformers)
        |
        v
   FAISS vector index  <-- persisted in vector_store/
        |
        v
User question --> embed --> retrieve top-k chunks --> Claude generates
                                                        grounded answer
```

## Project structure

```
rag_chatbot/
├── app.py              # Flask web app (chat UI + API)
├── rag_engine.py        # Core RAG pipeline: ingest, embed, retrieve, generate
├── ingest.py             # CLI to (re)build the vector index
├── requirements.txt
├── data/sample_docs/      # Sample documents to try the pipeline on
├── vector_store/           # FAISS index + metadata (created after ingestion)
├── templates/index.html
└── static/style.css
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Set your Anthropic API key (used for generation):
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```
   Or create a `.env` file in this folder with:
   ```
   ANTHROPIC_API_KEY=your-key-here
   ```
   Without a key, the app still runs and shows the raw retrieved context
   instead of a generated answer, so the retrieval half of the pipeline is
   demonstrable even offline.

3. Add your own documents to `data/sample_docs/` (or any folder), or use the
   two sample files already included.

4. Build the vector index:
   ```bash
   python ingest.py --folder data/sample_docs
   ```

5. Run the app:
   ```bash
   python app.py
   ```
   Open http://localhost:5000 in your browser.

## Notes for extending this project

- **Swap the embedding model**: change `EMBEDDING_MODEL_NAME` in
  `rag_engine.py`. Larger models (e.g. `all-mpnet-base-v2`) trade speed for
  retrieval quality.
- **Swap the vector store**: replace the FAISS calls in `RAGEngine` with a
  hosted store (Pinecone, Weaviate, Chroma) if you need persistence across
  machines or larger scale.
- **Chunking strategy**: the current chunker is fixed-size with overlap.
  A recursive/semantic chunker (splitting on paragraphs/sentences first)
  usually improves retrieval quality for structured documents.
- **Evaluation**: for a portfolio writeup, consider reporting retrieval
  precision@k on a small hand-labeled question set, plus a few example
  chat transcripts with sources.
