# Chat with Your Data — Streamlit RAG demo

Features:
- Upload PDFs, CSVs, or paste YouTube URLs to ingest content.
- Index documents into ChromaDB and query them using semantic search.
- Use LlamaIndex (optional) + ChromaDB for retrieval.
- Pluggable LLM/embeddings (Mistral/Groq/HF/OpenAI) via environment variables.
- Streamlit UI shows answers, confidence score, and highlighted source paragraphs.

Quickstart:
1. Copy `.env.example` to `.env` and fill your API keys if using remote models.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`
4. Upload files or add a YouTube URL, click "Ingest", then query.

Env vars:
- CHROMA_PERSIST_DIR (default: ./chroma_db)
- EMBEDDING_BACKEND (default: local_sentence_transformer)
- LLM_BACKEND (mistral | groq | hf | openai | local)
- HF_API_TOKEN, OPENAI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY — set as needed.

Notes:
- This scaffold uses sentence-transformers locally by default for speed & privacy.
- To switch to Mistral/Groq for LLM, supply the appropriate API key and set LLM_BACKEND.
