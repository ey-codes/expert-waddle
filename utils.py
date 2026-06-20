import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import requests

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local_sentence_transformer")
LLM_BACKEND = os.getenv("LLM_BACKEND", "local")
# load local embedding model
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
EMB_MODEL = None
if EMBEDDING_BACKEND == "local_sentence_transformer":
    EMB_MODEL = SentenceTransformer(EMB_MODEL_NAME)

def get_embeddings(texts):
    if EMBEDDING_BACKEND == "local_sentence_transformer":
        embs = EMB_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embs
    else:
        # placeholder for remote embedding endpoints (Mistral/Groq/HF)
        # Expect environment variables for endpoint + key
        raise NotImplementedError("Remote embedding backends are not implemented in scaffold. Use local_sentence_transformer.")

def get_chroma_client(persist_dir=CHROMA_DIR):
    return chromadb.Client(Settings(persist_directory=persist_dir, chroma_db_impl="duckdb+parquet"))

def upsert_to_chroma(docs, persist_dir=CHROMA_DIR, collection_name="documents"):
    client = get_chroma_client(persist_dir)
    col = None
    try:
        col = client.get_collection(collection_name)
    except Exception:
        col = client.create_collection(collection_name)
    texts = [d["text"] for d in docs]
    metadatas = [d.get("meta", {}) for d in docs]
    ids = [d["doc_id"] for d in docs]
    embeddings = get_embeddings(texts)
    col.upsert(ids=ids, metadatas=metadatas, documents=texts, embeddings=embeddings.tolist())

def list_collections(persist_dir=CHROMA_DIR):
    client = get_chroma_client(persist_dir)
    return client.list_collections()

def reset_chroma(persist_dir=CHROMA_DIR):
    # remove directory or re-create client & drop collections
    client = get_chroma_client(persist_dir)
    for c in client.list_collections():
        try:
            client.delete_collection(c["name"])
        except Exception:
            pass

class QueryEngine:
    def __init__(self, persist_dir=CHROMA_DIR, top_k=4, llm_backend=LLM_BACKEND):
        self.client = get_chroma_client(persist_dir)
        self.col = self.client.get_collection("documents")
        self.top_k = top_k
        self.llm_backend = llm_backend

    def _retrieve(self, query):
        q_emb = get_embeddings([query])[0]
        results = self.col.query(
            query_embeddings=[q_emb.tolist()],
            n_results=self.top_k,
            include=["metadatas", "documents", "distances"]
        )
        docs = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            # Chroma returns distance — for embeddings it's cosine by default (0..2). Convert to similarity
            score = 1 - dist
            docs.append({"text": doc, "meta": meta, "score": score})
        return docs

    def _call_llm(self, prompt):
        # Basic local/generic LLM wrapper. For remote models swap this to call Mistral/Groq/HF
        if self.llm_backend in ("openai", "hf", "mistral", "groq"):
            # Implement remote calls here using corresponding API token and endpoint
            # Placeholder: echo back prompt (replace with real API)
            return {"text": "LLM backend not configured in scaffold. Replace _call_llm with remote API call.", "confidence": None}
        else:
            # local simple answer: combine top chunks and return minimal summary
            combined = "\n\n".join([d["text"] for d in self.context])
            # naive extractive approach: return first 800 chars
            answer = combined[:1200] + ("..." if len(combined) > 1200 else "")
            return {"text": answer, "confidence": 0.5}

    def answer(self, query):
        self.context = self._retrieve(query)
        context_texts = [c["text"] for c in self.context]
        context_meta = [c["meta"] for c in self.context]
        prompt = f"Use the following context to answer the question. Context:\n\n{chr(10).join(context_texts)}\n\nQuestion: {query}\n\nAnswer concisely and list which chunks you used."
        llm_out = self._call_llm(prompt)
        # compute similarity-based confidence (max score)
        similarity_confidence = max([c["score"] for c in self.context]) if self.context else 0.0
        sources = []
        for c in self.context:
            sources.append({
                "doc_id": c["meta"].get("source", "unknown"),
                "text": c["text"],
                "score": c["score"]
            })
        return {
            "answer": llm_out["text"],
            "llm_confidence": llm_out.get("confidence"),
            "similarity_confidence": float(similarity_confidence),
            "sources": sources
        }
