# Chat with Your Data — Streamlit RAG demo (Gemini-ready)

This branch finalizes the Gemini-backed portfolio demo: improved UI, provenance, sample data, demo assets, and deployment instructions.

Features added in this branch
- Gemini (Google Generative Language) LLM integration (reads GEMINI_API_KEY from env or repo secrets)
- LLM instructed to return JSON with answer, provenance (indices into retrieved chunks), and confidence (0-1). Parser maps indices back to chunk metadata so the UI shows exact paragraph provenance.
- UI polish: chat history, highlighted query terms, links to sample_data files, copy/export transcript, and toggles to show full chunks.
- Sample data: 3 small CSV files in /sample_data for quick demo.
- Demo assets: one-minute demo script and screenshot/GIF instructions in /demo.
- Dockerfile included; deployment instructions for Streamlit Cloud / Render in README.

Quickstart
1. Add your Gemini key to GitHub repo secrets (recommended) or create a local `.env` (do NOT commit `.env`):
   - Name: `GEMINI_API_KEY`
   - Optionally: `GEMINI_MODEL` (default: text-bison-001)
   - Set `LLM_BACKEND=gemini` in `.env` or select in the UI.
2. Install deps: `pip install -r requirements.txt`
3. Run locally: `streamlit run app.py`

Security & cost notes
- Rotate the key and add project-level budget alerts in Google Cloud. Use conservative `max_output_tokens` and `temperature` defaults.

