"""Simple automated tests for common weather queries.
Sends three queries to the local ai-assistant and saves results to scripts/weather_test_results.json
Usage:
  python scripts/test_weather_queries.py
"""
import requests
import time
import json
import os

AI_URL = os.getenv('AI_ASSISTANT_URL', 'http://localhost:8000')
TIMEOUT = 25
OUTPATH = os.path.join(os.path.dirname(__file__), 'weather_test_results.json')

QUERIES = [
    # 1) Yesterday's weather in HCM
    {
        'id': 'yesterday_hcm',
        'text': 'Hôm qua thời tiết ở Hồ Chí Minh thế nào?',
        'type': 'weather',
    },
    # 2) Will it rain today in HCM?
    {
        'id': 'today_rain_hcm',
        'text': 'Hôm nay thời tiết Hồ Chí Minh có mưa không?',
        'type': 'weather',
    },
    # 3) Will it rain tomorrow in Ho Chi Minh?
    {
        'id': 'tomorrow_rain_hcm',
        'text': 'Ngày mai thời tiết Hồ Chí Minh có mưa không?',
        'type': 'weather',
    },
]


def call_save_clean_for_city(city: str):
    url = f"{AI_URL}/weather/save-clean?city={requests.utils.quote(city)}"
    return requests.post(url, timeout=TIMEOUT)


def call_ask(text: str):
    url = f"{AI_URL}/ask"
    payload = {'message': {'text': text}}
    return requests.post(url, json=payload, timeout=TIMEOUT)


def run():
    results = []
    for q in QUERIES:
        rec = {
            'id': q['id'],
            'text': q['text'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': None,
            'elapsed_s': None,
            'summary': None,
            'response': None,
        }
        start = time.time()
        try:
            # Use /ask so the normal pipeline is exercised; backend will call /weather/save-clean internally for weather intent
            r = call_ask(q['text'])
            rec['status'] = r.status_code
            elapsed = time.time() - start
            rec['elapsed_s'] = round(elapsed, 2)
            try:
                j = r.json()
                rec['response'] = j
                # Try to summarize meaningful fields
                if isinstance(j, dict):
                    if j.get('cleaned'):
                        cleaned = j['cleaned']
                        w = cleaned.get('weather', {}) or {}
                        a = cleaned.get('aqi', {}) or {}
                        rec['summary'] = f"cleaned: temp={w.get('temp')} aqi={a.get('aqi')} pm25={a.get('pm25')}"
                    else:
                        # prefer reply text
                        reply = j.get('reply') or j.get('response') or j.get('message')
                        rec['summary'] = (reply[:300] if isinstance(reply, str) else str(reply))
                else:
                    rec['summary'] = str(j)[:300]
            except Exception as e:
                rec['response'] = r.text
                rec['summary'] = r.text[:300]
        except Exception as e:
            rec['status'] = 'ERROR'
            rec['elapsed_s'] = round(time.time() - start, 2)
            rec['summary'] = str(e)
        print(f"[{rec['id']}] status={rec['status']} time={rec['elapsed_s']}s -> {rec['summary']}")
        results.append(rec)

    with open(OUTPATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('Saved results to', OUTPATH)


if __name__ == '__main__':
    run()
