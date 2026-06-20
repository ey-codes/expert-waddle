import os
import fitz  # PyMuPDF
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from utils import get_embeddings, upsert_to_chroma
from tqdm import tqdm
import io

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    start = 0
    chunks = []
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap
    return chunks

def ingest_pdf_file_bytes(file_bytes, file_name, persist_dir="./chroma_db"):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    docs = []
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        text = page.get_text()
        if text.strip():
            chunks = chunk_text(text)
            for i, c in enumerate(chunks):
                docs.append({
                    "doc_id": f"{file_name}::p{page_idx}::c{i}",
                    "text": c,
                    "meta": {"source": file_name, "page": page_idx, "chunk": i}
                })
    if docs:
        upsert_to_chroma(docs, persist_dir=persist_dir)

def ingest_file(uploaded_file, persist_dir="./chroma_db"):
    # uploaded_file is Streamlit UploadFile-like object
    name = uploaded_file.name
    b = uploaded_file.read()
    if name.lower().endswith(".pdf"):
        ingest_pdf_file_bytes(b, name, persist_dir=persist_dir)
    elif name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(b))
        for idx, row in df.iterrows():
            text = row.astype(str).to_csv(sep=" | ")
            docs = []
            for i, chunk in enumerate(chunk_text(text)):
                docs.append({
                    "doc_id": f"{name}::r{idx}::c{i}",
                    "text": chunk,
                    "meta": {"source": name, "row": int(idx), "chunk": i}
                })
            upsert_to_chroma(docs, persist_dir=persist_dir)
    else:
        # treat as text
        text = b.decode("utf-8", errors="ignore")
        docs = []
        for i, chunk in enumerate(chunk_text(text)):
            docs.append({
                "doc_id": f"{name}::c{i}",
                "text": chunk,
                "meta": {"source": name, "chunk": i}
            })
        upsert_to_chroma(docs, persist_dir=persist_dir)

def ingest_youtube(url, persist_dir="./chroma_db"):
    # extract video id
    if "v=" in url:
        vid = url.split("v=")[1].split("&")[0]
    else:
        vid = url.rstrip("/").split("/")[-1]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid)
    except Exception as e:
        transcript = []
    text = " ".join([t["text"] for t in transcript])
    docs = []
    for i, chunk in enumerate(chunk_text(text)):
        docs.append({
            "doc_id": f"youtube::{vid}::c{i}",
            "text": chunk,
            "meta": {"source": f"youtube::{vid}", "chunk": i}
        })
    upsert_to_chroma(docs, persist_dir=persist_dir)
