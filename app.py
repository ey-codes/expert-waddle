import streamlit as st
from dotenv import load_dotenv
import os
from ingest import ingest_file, ingest_youtube
from utils import QueryEngine, list_collections, reset_chroma
from io import StringIO
import json
load_dotenv()

st.set_page_config(page_title="Chat with Your Data", layout="wide")
st.title("Chat with Your Data — Gemini RAG demo")

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

# Settings
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top K retrieval", 1, 10, 4)
backend = st.sidebar.selectbox("LLM backend", ["local", "gemini"]) 

# Chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Query area
st.header("Ask a question")
query = st.text_input("Enter your question")
if st.button("Ask") and query.strip():
    with st.spinner("Searching and generating answer..."):
        engine = QueryEngine(persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
                             top_k=top_k,
                             llm_backend=backend)
        result = engine.answer(query)
    st.session_state.history.append({"query": query, "result": result})

# Conversation pane
st.subheader("Conversation")
for i, turn in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**Q:** {turn['query']}")
    st.markdown(f"**A:** {turn['result']['answer']}")
    st.write(f"Similarity confidence: **{turn['result']['similarity_confidence']:.2f}**")
    if turn['result'].get('llm_confidence') is not None:
        st.write(f"LLM self-rated confidence: **{turn['result']['llm_confidence']}**")

    st.markdown("**Sources (provenance):**")
    for src in turn['result']['sources']:
        source = src.get('doc_id', 'unknown')
        link = None
        if source.startswith('sample_data/'):
            # point to main branch raw file
            link = f"https://github.com/ey-codes/expert-waddle/blob/main/{source}"
        if link:
            st.markdown(f"- **[{source}]({link})** (score: {src['score']:.3f})")
        else:
            st.markdown(f"- **{source}** (score: {src['score']:.3f})")
        snippet = src['text']
        # highlight query terms naively
        display_snippet = snippet
        for tok in set([t for t in query.split() if len(t) > 3]):
            display_snippet = display_snippet.replace(tok, f"**{tok}**")
        with st.expander("Show chunk"): 
            st.write(display_snippet)

# Export transcript
if st.button("Export transcript"):
    out = StringIO()
    for turn in st.session_state.history:
        out.write("Q: " + turn['query'] + "\n")
        out.write("A: " + turn['result']['answer'] + "\n")
        out.write("---\n")
    st.download_button("Download transcript", out.getvalue(), file_name="transcript.txt")

st.markdown("---")
st.caption("Tip: Use the sample data (folder /sample_data) for a quick demo. Set GEMINI_API_KEY in your environment.")
