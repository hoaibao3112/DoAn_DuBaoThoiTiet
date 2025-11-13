import json
import sys
from urllib import request

URL = "http://localhost:5678/zalo-webhook/zalo-ai-assistant"

payload = {
    "events": [
        {
            "type": "user_send_text",
            "sender": {"id": "123456789"},
            "message": {"text": "Thời tiết Hà Nội hôm nay?"},
        }
    ]
}

data = json.dumps(payload).encode('utf-8')
req = request.Request(URL, data=data, headers={'Content-Type': 'application/json'})
try:
    with request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode('utf-8')
        print('HTTP', resp.status)
        try:
            print(json.dumps(json.loads(body), ensure_ascii=False, indent=2))
        except Exception:
            print(body)
except Exception as e:
    print('Request failed:', e)
    sys.exit(1)
