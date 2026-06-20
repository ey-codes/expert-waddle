# Chat with Your Data — Streamlit RAG demo

Features:
- Upload PDFs, CSVs, or paste YouTube URLs to ingest content.
- Index documents into ChromaDB and query them using semantic search.
- Use LlamaIndex (optional) + ChromaDB for retrieval.
- Pluggable LLM/embeddings (Gemini / Mistral / Groq / HF / OpenAI / local) via environment variables.
- Streamlit UI shows answers, confidence score, and highlighted source paragraphs.

Quickstart:
1. Copy `.env.example` to `.env` and fill your API keys if using remote models.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`
4. Upload files or add a YouTube URL, click "Ingest", then ask questions.

Env vars (new for Gemini integration):
- GEMINI_API_KEY — your Google Generative Language API key (or other Gemini key). Do NOT commit this value.
- GEMINI_MODEL (optional) — model id to use (default: text-bison-001). Example: text-bison-001 or chat-bison-001
- If you prefer using a Google service account JSON, set GOOGLE_APPLICATION_CREDENTIALS to the JSON file path on the host.

Security notes:
- Add keys to GitHub repository secrets (Settings → Secrets and variables → Actions) or your host's environment configuration; never commit `.env` or keys to git.

How Gemini is used:
- The app reads GEMINI_API_KEY at runtime and calls the public Generative Language endpoint (v1beta2) to generate an answer from retrieved context chunks.
- If you use a service account instead, configure GOOGLE_APPLICATION_CREDENTIALS and the code will attempt to use OAuth to fetch an access token.

Switching backends:
- Set LLM_BACKEND in `.env` or select the backend in the Streamlit UI. Use `gemini` to enable the Gemini/Generative API integration.

Notes:
- This scaffold aims to be simple and clear for portfolio demonstration. Rotate and protect API keys; set budget alerts in Google Cloud to avoid unexpected costs.
