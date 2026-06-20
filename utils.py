import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import requests
import time

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local_sentence_transformer")
LLM_BACKEND = os.getenv("LLM_BACKEND", "local")
# load local embedding model
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
EMB_MODEL = None
if EMBEDDING_BACKEND == "local_sentence_transformer":
    EMB_MODEL = SentenceTransformer(EMB_MODEL_NAME)

# Gemini / Google Generative Language configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "text-bison-001")
GEMINI_ENDPOINT_BASE = "https://generativelanguage.googleapis.com/v1beta2/models"

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

def _call_gemini_api(prompt, max_retries=2, timeout=30):
    """
    Call Google's Generative Language API (v1beta2) using an API key.
    The function is resilient and will try a couple times on transient errors.
    Returns the generated text on success, or raises an exception.
    """
    api_key = GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    # Build URL
    model = GEMINI_MODEL
    url = f"{GEMINI_ENDPOINT_BASE}/{model}:generate?key={api_key}"

    payload = {
        "prompt": {"text": prompt},
        # tuning fields
        "temperature": 0.2,
        "candidate_count": 1,
        "max_output_tokens": 512
    }

    headers = {
        "Content-Type": "application/json"
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                # Parsing: generative language responses vary; handle common patterns
                # v1beta2 typically has 'candidates' -> list with 'output' or 'content' or 'text'
                if "candidates" in data and isinstance(data["candidates"], list) and len(data["candidates"])>0:
                    cand = data["candidates"][0]
                    # Try various keys
                    for k in ("output", "content", "text", "message"):
                        if k in cand:
                            return cand[k]
                    # Some responses embed the text in 'content' -> list of structured segments
                    if "content" in cand and isinstance(cand["content"], list):
                        # join any string segments
                        pieces = []
                        for seg in cand["content"]:
                            if isinstance(seg, dict) and "text" in seg:
                                pieces.append(seg["text"])
                            elif isinstance(seg, str):
                                pieces.append(seg)
                        if pieces:
                            return "".join(pieces)
                # fallback: try top-level 'output' or 'text'
                for k in ("output", "text", "response"):
                    if k in data:
                        return data[k]
                # if none matched, return the raw json as string (best-effort)
                return json.dumps(data)
            else:
                # Retry on 429/5xx; raise on 4xx client errors (except 429)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    time.sleep(1 + attempt * 2)
                    continue
                else:
                    raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
        except requests.RequestException as e:
            if attempt < max_retries:
                time.sleep(1 + attempt * 2)
                continue
            raise

class QueryEngine:
    def __init__(self, persist_dir=CHROMA_DIR, top_k=4, llm_backend=LLM_BACKEND):
        self.client = get_chroma_client(persist_dir)
        try:
            self.col = self.client.get_collection("documents")
        except Exception:
            # If collection missing, create it so queries won't fail later
            self.col = self.client.create_collection("documents")
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
        # results structure: documents -> list of lists
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs_list, metas_list, dists_list):
            # Chroma returns distance — for embeddings it's cosine by default (0..2). Convert to similarity
            try:
                score = 1 - float(dist)
            except Exception:
                score = 0.0
            docs.append({"text": doc, "meta": meta, "score": score})
        return docs

    def _call_llm(self, prompt):
        # Choose backend
        if self.llm_backend == "gemini":
            try:
                text = _call_gemini_api(prompt)
                return {"text": text, "confidence": None}
            except Exception as e:
                return {"text": f"[Gemini request failed] {e}", "confidence": 0.0}
        elif self.llm_backend in ("openai", "hf", "mistral", "groq"):
            # Not implemented in this branch — keep placeholder
            return {"text": "LLM backend not configured in scaffold for this provider. Provide an API key and implementation.", "confidence": None}
        else:
            # local simple answer: combine top chunks and return minimal summary
            combined = "\n\n".join([d["text"] for d in self.context])
            answer = combined[:1200] + ("..." if len(combined) > 1200 else "")
            return {"text": answer, "confidence": 0.5}

    def answer(self, query):
        self.context = self._retrieve(query)
        context_texts = [c["text"] for c in self.context]
        prompt = (
            "You are a helpful assistant. Use the following context to answer the question concisely. "
            "If the answer isn't contained in the context, say you don't know. Include short provenance references.\n\n"
            "Context:\n\n" + "\n\n".join(context_texts) + f"\n\nQuestion: {query}\n\nAnswer:" 
        )
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
