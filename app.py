import streamlit as st
from dotenv import load_dotenv
import os
from ingest import ingest_file, ingest_youtube
from utils import QueryEngine, list_collections, reset_chroma
load_dotenv()

st.set_page_config(page_title="Chat with Your Data", layout="wide")
st.title("Chat with Your Data — RAG demo")

# Sidebar: ingestion
st.sidebar.header("Ingest")
upload = st.sidebar.file_uploader("Upload PDF or CSV", accept_multiple_files=True)
youtube_url = st.sidebar.text_input("YouTube URL (optional)")
if st.sidebar.button("Ingest"):
    with st.spinner("Ingesting..."):
        for f in upload:
            ingest_file(f, persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
        if youtube_url:
            ingest_youtube(youtube_url, persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    st.sidebar.success("Ingestion complete.")

if st.sidebar.button("Reset/clear DB"):
    reset_chroma(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    st.sidebar.warning("Chroma DB cleared.")

# Main UI: Query
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top K retrieval", 1, 10, 4)
backend = st.sidebar.selectbox("LLM backend", ["local", "hf", "openai", "mistral", "groq"]) 
# Query area
st.header("Ask a question")
query = st.text_input("Enter your question")
if st.button("Ask") and query.strip():
    with st.spinner("Searching and generating answer..."):
        engine = QueryEngine(persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
                             top_k=top_k,
                             llm_backend=backend)
        result = engine.answer(query)
    # Show answer
    st.subheader("Answer")
    st.markdown(result["answer"])
    st.write(f"Confidence (similarity-based): **{result['similarity_confidence']:.2f}**")
    st.write(f"LLM self-evaluated confidence: **{result.get('llm_confidence','N/A')}**")
    st.markdown("---")
    st.subheader("Sources")
    for src in result["sources"]:
        st.markdown(f"**{src['doc_id']}** (score: {src['score']:.3f})")
        # show snippet and highlight query terms (simple)
        snippet = src["text"]
        st.code(snippet[:1000] + ("..." if len(snippet) > 1000 else ""))
