#!/usr/bin/env python3
"""
Simulate a Telegram update POST to an n8n webhook URL.
Usage:
  python scripts/simulate_telegram_webhook.py [webhook_url]
If no URL is provided it defaults to http://localhost:5678/webhook/telegram-webhook/telegram-ai-assistant
"""
import sys
import json
from urllib import request, error

DEFAULT = 'http://localhost:5678/webhook/telegram-webhook/telegram-ai-assistant'

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    payload = {
        "update_id": 100000000,
        "message": {
            "message_id": 1,
            "from": {"id": 111111111, "is_bot": False, "first_name": "TestUser"},
            "chat": {"id": 111111111, "type": "private"},
            "date": 1700000000,
            "text": "Thời tiết Hà Nội hôm nay?"
        }
    }
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method='POST')
    try:
        with request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode('utf-8')
            print('Response code:', resp.getcode())
            print('Response body:', body)
    except error.HTTPError as e:
        print('Request failed: HTTP Error', e.code)
        try:
            print(e.read().decode('utf-8'))
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print('Request failed:', e)
        sys.exit(1)

if __name__ == '__main__':
    main()
