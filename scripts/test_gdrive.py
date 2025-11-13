#!/usr/bin/env python3
"""
Quick test to verify Google Service Account can write to a Drive folder.

Usage (PowerShell):
  # ensure virtualenv is active and deps installed
  pip install google-api-python-client google-auth
  python .\scripts\test_gdrive.py

The script will look for:
 - secrets/service_account.json (default) or path from env var GOOGLE_APPLICATION_CREDENTIALS
 - GDRIVE_FOLDER_ID in .env or environment

"""
import os
import sys
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build


def load_env_dotenv(path='.env'):
    env = {}
    if not os.path.exists(path):
        return env
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip()
    return env


def main():
    env = load_env_dotenv()
    key_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') or env.get('GOOGLE_APPLICATION_CREDENTIALS') or 'secrets/service_account.json'
    folder_id = os.environ.get('GDRIVE_FOLDER_ID') or env.get('GDRIVE_FOLDER_ID')

    print('Using key file:', key_path)
    print('Using GDRIVE_FOLDER_ID:', folder_id)

    if not os.path.exists(key_path):
        print('\nERROR: service account key file not found at', key_path)
        print('Place your downloaded JSON key at that path and try again.')
        sys.exit(2)

    if not folder_id:
        print('\nERROR: GDRIVE_FOLDER_ID not set in environment or .env')
        print('Set GDRIVE_FOLDER_ID in your .env (folder ID from Drive URL) and try again.')
        sys.exit(3)

    scopes = ['https://www.googleapis.com/auth/drive.file']
    try:
        creds = service_account.Credentials.from_service_account_file(key_path, scopes=scopes)
        service = build('drive', 'v3', credentials=creds)

        # Try creating a small test file in the folder
        file_metadata = {
            'name': 'doan_test_from_service_account.txt',
            'parents': [folder_id]
        }
        created = service.files().create(body=file_metadata, fields='id, name').execute()
        print('\nSUCCESS: created file in folder:')
        print(json.dumps(created, indent=2))
        print('\nYou can check the Drive folder for "doan_test_from_service_account.txt"')

    except Exception as e:
        print('\nERROR during Drive API call:')
        print(type(e).__name__, str(e))
        print('\nCommon causes:')
        print('- service account key path incorrect')
        print('- folder not shared with service account email')
        print('- insufficient permissions (share folder or grant roles)')
        sys.exit(4)


if __name__ == '__main__':
    main()
