from dotenv import load_dotenv
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import requests
import time

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local_sentence_transformer")
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")
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
    client = get_chroma_client(persist_dir)
    for c in client.list_collections():
        try:
            client.delete_collection(c["name"])
        except Exception:
            pass


def _call_gemini_api(prompt, max_retries=2, timeout=30):
    api_key = GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    model = GEMINI_MODEL
    url = f"{GEMINI_ENDPOINT_BASE}/{model}:generate?key={api_key}"
    payload = {
        "prompt": {"text": prompt},
        "temperature": 0.2,
        "candidate_count": 1,
        "max_output_tokens": 512
    }
    headers = {"Content-Type": "application/json"}
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if "candidates" in data and isinstance(data["candidates"], list) and len(data["candidates"])>0:
                    cand = data["candidates"][0]
                    for k in ("output", "content", "text", "message"):
                        if k in cand:
                            # return string content if present
                            val = cand[k]
                            if isinstance(val, dict):
                                return json.dumps(val)
                            return val
                    if "content" in cand and isinstance(cand["content"], list):
                        pieces = []
                        for seg in cand["content"]:
                            if isinstance(seg, dict) and "text" in seg:
                                pieces.append(seg["text"])
                            elif isinstance(seg, str):
                                pieces.append(seg)
                        if pieces:
                            return "".join(pieces)
                for k in ("output", "text", "response"):
                    if k in data:
                        return data[k]
                return json.dumps(data)
            else:
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


def _parse_json_from_text(text):
    import re
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None

class QueryEngine:
    def __init__(self, persist_dir=CHROMA_DIR, top_k=4, llm_backend=LLM_BACKEND):
        self.client = get_chroma_client(persist_dir)
        try:
            self.col = self.client.get_collection("documents")
        except Exception:
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
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs_list, metas_list, dists_list):
            try:
                score = 1 - float(dist)
            except Exception:
                score = 0.0
            docs.append({"text": doc, "meta": meta, "score": score})
        return docs

    def _call_llm(self, prompt):
        if self.llm_backend == "gemini":
            try:
                raw = _call_gemini_api(prompt)
                parsed = _parse_json_from_text(raw)
                if parsed:
                    return {"text": parsed.get('answer', raw), "provenance": parsed.get('provenance'), "confidence": parsed.get('confidence')}
                else:
                    return {"text": raw, "provenance": None, "confidence": None}
            except Exception as e:
                return {"text": f"[Gemini request failed] {e}", "provenance": None, "confidence": 0.0}
        else:
            combined = "\n\n".join([d["text"] for d in self.context])
            answer = combined[:1200] + ("..." if len(combined) > 1200 else "")
            return {"text": answer, "provenance": None, "confidence": 0.5}

    def answer(self, query):
        self.context = self._retrieve(query)
        context_texts = [f"[{i}] {c['text']}" for i, c in enumerate(self.context)]
        prompt = (
            "You are a helpful assistant. Use the following context to answer the question concisely. "
            "If the answer isn't contained in the context, say you don't know. At the end, return a JSON object with fields: answer (string), provenance (list of indices into the context used), and confidence (0-1).\n\n"
            "Context:\n\n" + "\n\n".join(context_texts) + f"\n\nQuestion: {query}\n\nAnswer:" 
        )
        llm_out = self._call_llm(prompt)
        similarity_confidence = max([c["score"] for c in self.context]) if self.context else 0.0
        sources = []
        for idx, c in enumerate(self.context):
            sources.append({
                "doc_id": c["meta"].get("source", "unknown"),
                "text": c["text"],
                "score": c["score"],
                "index": idx
            })
        prov = llm_out.get('provenance')
        prov_sources = []
        if isinstance(prov, list):
            for p in prov:
                if isinstance(p, int) and 0 <= p < len(sources):
                    prov_sources.append(sources[p])
        return {
            "answer": llm_out.get("text"),
            "llm_confidence": llm_out.get("confidence"),
            "similarity_confidence": float(similarity_confidence),
            "sources": prov_sources if prov_sources else sources
        }
