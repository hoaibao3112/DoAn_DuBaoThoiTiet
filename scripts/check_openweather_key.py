import os
from pathlib import Path
import requests
import json

root = Path(__file__).resolve().parents[1]
env_file = root / '.env'
key = None
base = None
if env_file.exists():
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip()
                if k == 'OPENWEATHER_API_KEY':
                    key = v
                if k == 'OPENWEATHER_BASE_URL':
                    base = v
    except Exception as e:
        print('ERROR_READING_.env', e)

# fallback to environment
if not key:
    key = os.getenv('OPENWEATHER_API_KEY')
if not base:
    base = os.getenv('OPENWEATHER_BASE_URL', 'https://api.openweathermap.org/data/2.5')

if not key:
    print('NO_KEY')
    raise SystemExit(2)

if not base:
    base = 'https://api.openweathermap.org/data/2.5'

url = base.rstrip('/') + '/forecast'
params = {'q': 'Hanoi', 'units': 'metric', 'appid': key}
print('Calling:', url)
try:
    r = requests.get(url, params=params, timeout=10)
    print('HTTP', r.status_code)
    try:
        j = r.json()
    except Exception as ex:
        print('INVALID_JSON', ex)
        print(r.text[:1000])
        raise SystemExit(3)

    # quick checks
    if r.status_code == 200 and isinstance(j, dict):
        lst = j.get('list')
        city = j.get('city', {})
        print('city:', city.get('name'), city.get('country'))
        if isinstance(lst, list):
            print('list_count:', len(lst))
            if len(lst) > 0:
                first = lst[0]
                print('first.dt_txt:', first.get('dt_txt'))
                main = first.get('main', {})
                print('first.temp:', main.get('temp'))
                rain = first.get('rain') or {}
                snow = first.get('snow') or {}
                print('first.rain_3h:', rain.get('3h'))
                print('first.snow_3h:', snow.get('3h'))
            print('\nKEY_OK')
        else:
            print('NO_LIST_IN_RESPONSE')
            print(json.dumps(j)[:1000])
    else:
        # API returned non-200
        print('API_ERROR')
        print(json.dumps(j)[:1000])
        raise SystemExit(4)
except requests.exceptions.RequestException as e:
    print('REQUEST_EXCEPTION', e)
    raise
