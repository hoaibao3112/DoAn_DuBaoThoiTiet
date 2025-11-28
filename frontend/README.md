Streamlit frontend for the AI Assistant project

Quick start

1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

2. Run Streamlit (local)

```powershell
streamlit run frontend/streamlit_app.py
```

3. Default configuration

- By default the frontend sends requests to `http://localhost:8000/ask`.
- If `ai-assistant` runs in Docker, either add the `streamlit` service to `docker-compose.yml` or run Streamlit locally and set `AI_ASSISTANT_URL` in the sidebar.

Notes

- To save results to Google Drive, ensure `GDRIVE_FOLDER_ID` and `GOOGLE_APPLICATION_CREDENTIALS` are configured in the environment where `ai-assistant` runs.
