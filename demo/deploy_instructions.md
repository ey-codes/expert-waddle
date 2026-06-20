# Deploying to Streamlit Cloud or Render

Streamlit Cloud (quick):
1. Go to https://streamlit.io/cloud and sign in with GitHub.
2. Click "New app" and select the repo `ey-codes/expert-waddle`, branch `gemini-complete`.
3. In the Streamlit app settings, add the environment variable GEMINI_API_KEY (do NOT paste it into code).
4. Deploy. The app will run the same code as locally.

Render / Railway (Docker):
1. Create a new service and connect your GitHub repo.
2. Use the Dockerfile at the repo root; set the start command: `streamlit run app.py`.
3. Add environment variable GEMINI_API_KEY in the service settings.
4. Deploy and monitor logs.

Notes:
- For both hosts, set budget alerts on your Google Cloud project.
- Ensure you never commit `.env` or API keys to the repository.
