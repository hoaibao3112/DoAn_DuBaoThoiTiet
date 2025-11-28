#!/usr/bin/env python3
import os,sys,json
from urllib import request, error

# Read .env
env={}
with open('.env','r',encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k,v=line.split('=',1)
        env[k.strip()]=v.strip()
user=env.get('N8N_USER')
pwd=env.get('N8N_PASSWORD')
if not user or not pwd:
    print('Missing N8N_USER/N8N_PASSWORD in .env')
    sys.exit(1)

# Workflow IDs to try activating (from logs)
workflow_ids=['1GLdjMaKcqoBlmaj','a1an0ndTjWRFYh8j','VsApURKHsdanoKqT','pvO9rVilgLFQpNin','i0CeIpSmbjMmblqU','xGhXLeBJ67l1TJwo']

# login
login_url='http://localhost:5678/login'
login_payload=json.dumps({'email':user,'password':pwd,'emailOrLdapLoginId':user}).encode('utf-8')
req=request.Request(login_url,data=login_payload,headers={'Content-Type':'application/json'},method='POST')
try:
    with request.urlopen(req,timeout=10) as resp:
        cookie=resp.getheader('Set-Cookie')
        if not cookie:
            print('Login succeeded but no Set-Cookie returned')
            sys.exit(1)
        print('Login OK')
except Exception as e:
    print('Login failed:',e)
    sys.exit(1)

for wid in workflow_ids:
    act_url=f'http://localhost:5678/rest/workflows/{wid}/activate'
    req2=request.Request(act_url,data=b'',headers={'Content-Type':'application/json','Cookie':cookie},method='POST')
    try:
        with request.urlopen(req2,timeout=10) as r:
            body=r.read().decode('utf-8')
            print('\nActivated',wid,'status',r.getcode())
            print(body[:2000])
    except error.HTTPError as e:
        print('\nActivate failed for',wid,'HTTP',e.code)
        try:
            print(e.read().decode())
        except:
            pass
    except Exception as e:
        print('\nActivate error for',wid,':',e)

print('\nDone')
