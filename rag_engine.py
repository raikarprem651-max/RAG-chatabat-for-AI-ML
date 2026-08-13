"""
rag_engine.py
--------------
Core Retrieval-Augmented Generation (RAG) engine.

Pipeline:
1. Load documents (.txt, .pdf, .docx) from a folder.
2. Split them into overlapping chunks.
3. Embed chunks with a sentence-transformers model.
4. Store embeddings in a FAISS vector index (persisted to disk).
5. At query time: embed the question, retrieve top-k similar chunks,
   and pass them as context to an LLM (Anthropic Claude) to generate
   a grounded answer with source citations.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import anthropic
except ImportError:
    anthropic = None


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good quality
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 80      # overlap between consecutive chunks
TOP_K = 4                # number of chunks retrieved per query

INDEX_DIR = Path(__file__).parent / "vector_store"
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "meta.pkl"


# --------------------------------------------------------------------------
# Document loading
# --------------------------------------------------------------------------

def load_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt" or suffix == ".md":
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf not installed — run pip install pypdf")
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        if DocxDocument is None:
            raise RuntimeError("python-docx not installed — run pip install python-docx")
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# --------------------------------------------------------------------------
# Main RAG engine class
# --------------------------------------------------------------------------

class RAGEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index = None
        self.metadata: List[Dict] = []   # parallel list: {"text":..., "source":...}
        self._load_index_if_exists()

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key) if (anthropic and api_key) else None

    # ---------------- Index build / persistence ----------------

    def _load_index_if_exists(self):
        if INDEX_FILE.exists() and META_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))
            with open(META_FILE, "rb") as f:
                self.metadata = pickle.load(f)

    def _save_index(self):
        INDEX_DIR.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(META_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

    def build_index_from_folder(self, folder: str):
        """Ingest every supported file in `folder`, chunk it, embed it, and build a fresh FAISS index."""
        folder_path = Path(folder)
        files = [p for p in folder_path.rglob("*") if p.suffix.lower() in (".txt", ".md", ".pdf", ".docx")]
        if not files:
            raise ValueError(f"No supported documents found in {folder}")

        all_chunks, all_meta = [], []
        for path in tqdm(files, desc="Loading documents"):
            text = load_text_from_file(path)
            chunks = chunk_text(text)
            for c in chunks:
                all_chunks.append(c)
                all_meta.append({"text": c, "source": path.name})

        embeddings = self.embedder.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(self.dim)  # cosine similarity via normalized inner product
        index.add(embeddings.astype(np.float32))

        self.index = index
        self.metadata = all_meta
        self._save_index()
        return len(all_chunks)

    # ---------------- Retrieval ----------------

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({"text": meta["text"], "source": meta["source"], "score": float(score)})
        return results

    # ---------------- Generation ----------------

    def answer(self, query: str, top_k: int = TOP_K, history: List[Dict] = None) -> Dict:
        context_chunks = self.retrieve(query, top_k=top_k)

        if not context_chunks:
            return {
                "answer": "I don't have any documents indexed yet. Please add files to the data folder and run ingestion first.",
                "sources": [],
            }

        context_text = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
        )

        system_prompt = (
            "You are a helpful assistant that answers questions using ONLY the provided "
            "context. If the answer is not contained in the context, say you don't know "
            "based on the available documents. Cite the source filename(s) you used in "
            "your answer."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        if self.client is None:
            # No API key configured — return the retrieved context directly so the
            # retrieval half of the pipeline is still demonstrable end-to-end.
            return {
                "answer": (
                    "[No ANTHROPIC_API_KEY set — showing retrieved context instead of a "
                    "generated answer]\n\n" + context_text
                ),
                "sources": [c["source"] for c in context_chunks],
            }

        messages = (history or []) + [{"role": "user", "content": user_prompt}]

        response = self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            system=system_prompt,
            messages=messages,
        )
        answer_text = "".join(block.text for block in response.content if block.type == "text")

        return {
            "answer": answer_text,
            "sources": list({c["source"] for c in context_chunks}),
        }
