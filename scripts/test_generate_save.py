import os
import json

# Lightweight .env loader (avoid dependency on python-dotenv)
def load_dotenv_file(path='.env'):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            # Do not overwrite existing environment variables
            if k not in os.environ:
                os.environ[k] = v

load_dotenv_file()

# Ensure GOOGLE_APPLICATION_CREDENTIALS and GDRIVE_FOLDER_ID are present
sa = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
folder = os.getenv("GDRIVE_FOLDER_ID")
print("Using service account:", sa)
print("Using GDRIVE_FOLDER_ID:", folder)

if not sa or not folder:
    raise SystemExit("Set GOOGLE_APPLICATION_CREDENTIALS and GDRIVE_FOLDER_ID in .env before running this test.")

import sys
from pathlib import Path
# ensure project root is on sys.path so `app` package can be imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import generate
from app.schemas import GenerateRequest


def run_test():
    payload = GenerateRequest(prompt="Báo cáo thử nghiệm lưu báo cáo thời tiết - test", user_id=None)
    print("Calling generate() handler...\n")
    resp = generate(payload)
    # resp is a pydantic model; convert to dict
    try:
        d = resp.dict()
    except Exception:
        # If it's returned as object with attributes
        d = {
            'success': getattr(resp, 'success', None),
            'ad_text': getattr(resp, 'ad_text', None),
            'drive_file_id': getattr(resp, 'drive_file_id', None),
        }

    print("Response:")
    print(json.dumps(d, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run_test()
