#!/usr/bin/env python3
import base64, json, sys
from urllib import request, error

def read_env(path='.env'):
    d={}
    try:
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k,v=line.split('=',1)
                d[k.strip()]=v.strip()
    except Exception as e:
        print('Failed reading .env:', e)
    return d


def get(url,user,pwd):
    auth = base64.b64encode(f"{user}:{pwd}".encode()).decode()
    req = request.Request(url, headers={'Authorization':'Basic '+auth})
    try:
        with request.urlopen(req, timeout=10) as resp:
            b=resp.read().decode()
            return resp.getcode(), b
    except error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return None, str(e)


env=read_env()
user=env.get('N8N_USER')
pwd=env.get('N8N_PASSWORD')
if not user or not pwd:
    print('Missing N8N_USER/N8N_PASSWORD in .env')
    sys.exit(1)

url='http://localhost:5678/rest/workflows'
code,body = get(url,user,pwd)
print('Status:',code)
try:
    data=json.loads(body)
    for wf in data:
        print('ID:',wf.get('id'))
        print('Name:',wf.get('name'))
        print('Active:',wf.get('active'))
        print('---')
except Exception:
    print(body)
