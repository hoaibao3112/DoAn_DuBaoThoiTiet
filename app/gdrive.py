import os
import json
import datetime
import io

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


def save_json_to_drive(data: dict, folder_id: str) -> str:
    """Save a JSON object as a file into Google Drive folder using a service account.

    Returns the created file id.
    """
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not sa_path:
        raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS is not set")
    if not os.path.exists(sa_path):
        raise FileNotFoundError(f"service account file not found at {sa_path}")

    scopes = [
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(sa_path, scopes=scopes)
    service = build("drive", "v3", credentials=creds)

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"dulieu_{ts}.json"

    body = io.BytesIO(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    media = MediaIoBaseUpload(body, mimetype="application/json")

    metadata = {"name": filename, "parents": [folder_id]}
    created = service.files().create(body=metadata, media_body=media, fields="id").execute()
    return created.get("id")
