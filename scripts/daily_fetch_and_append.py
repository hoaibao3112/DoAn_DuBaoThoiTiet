"""
Simple script to call the server `/weather/save-clean` endpoint for a list of cities
and thereby append cleaned daily records to `data/cleaned/station_day_clean.csv`.

Configure via environment variables or edit the `CITIES` list below.

Usage:
  python scripts/daily_fetch_and_append.py

Exit codes:
  0 success
  2 no AI_ASSISTANT_URL

"""
import os
import time
import requests
from pathlib import Path

AI_URL = os.getenv('AI_ASSISTANT_URL') or 'http://localhost:8000/ask'
if not AI_URL:
    print('AI_ASSISTANT_URL not set; set it in .env or environment')
    raise SystemExit(2)

SAVE_CLEAN_URL = AI_URL.replace('/ask', '/weather/save-clean')
# Edit this list to the cities you want to collect daily. You can also set CITIES env var as comma-separated.
CITIES = os.getenv('DAILY_CITIES', 'Hanoi,Ho Chi Minh').split(',')
CITIES = [c.strip() for c in CITIES if c.strip()]
TIMEOUT = int(os.getenv('FETCH_TIMEOUT', '15'))
PAUSE_BETWEEN = float(os.getenv('PAUSE_BETWEEN', '1.0'))

LOG_PATH = Path(__file__).resolve().parents[1] / 'logs'
LOG_PATH.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_PATH / f'daily_fetch_{time.strftime()%Y%m%d}.log' if False else LOG_PATH / 'daily_fetch.log'


def fetch_and_log(city: str) -> bool:
    url = f"{SAVE_CLEAN_URL}?city={requests.utils.quote(city)}"
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        r = requests.post(url, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        ok = j.get('success') is True
        print(f"[{now}] {city}: HTTP {r.status_code} success={ok}")
        return ok
    except Exception as e:
        print(f"[{now}] {city}: ERROR {e}")
        return False


def main():
    success_count = 0
    for city in CITIES:
        ok = fetch_and_log(city)
        if ok:
            success_count += 1
        time.sleep(PAUSE_BETWEEN)
    print(f"Done: {success_count}/{len(CITIES)} succeeded.")


if __name__ == '__main__':
    main()
