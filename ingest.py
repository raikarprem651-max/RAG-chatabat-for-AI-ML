"""
ingest.py
---------
CLI to (re)build the FAISS vector index from documents in a folder.

Usage:
    python ingest.py --folder data/sample_docs
"""

import argparse
from rag_engine import RAGEngine


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    parser.add_argument("--folder", type=str, default="data/sample_docs",
                         help="Folder containing .txt, .md, .pdf, or .docx files")
    args = parser.parse_args()

    engine = RAGEngine()
    n_chunks = engine.build_index_from_folder(args.folder)
    print(f"Indexed {n_chunks} chunks from '{args.folder}'. Vector store saved to vector_store/.")


if __name__ == "__main__":
    main()
