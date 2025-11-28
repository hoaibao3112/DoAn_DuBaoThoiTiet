#!/usr/bin/env python3
"""
Import and activate an n8n workflow via the REST API using Basic Auth.
Usage: python scripts/import_activate_n8n.py [workflow_json_path]
Reads N8N_USER and N8N_PASSWORD from .env in the repo root.
"""
import sys, base64, json
from urllib import request, error

ENV_PATH = '.env'
DEFAULT_JSON = 'n8n-workflows/Telegram_AI_Assistant.json'

def read_env(path):
    d = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k,v=line.split('=',1)
                d[k.strip()]=v.strip()
    except Exception as e:
        print('Failed reading .env:', e)
    return d


def post_json(url, data_bytes, user, pwd):
    # Try Basic Auth first
    auth = base64.b64encode(f"{user}:{pwd}".encode('utf-8')).decode('ascii')
    req = request.Request(url, data=data_bytes, headers={
        'Content-Type': 'application/json',
        'Authorization': 'Basic ' + auth
    }, method='POST')
    try:
        with request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode('utf-8')
            return resp.getcode(), body, None
    except error.HTTPError as e:
        try:
            body = e.read().decode('utf-8')
        except Exception:
            body = ''
        code = e.code
        # If Basic Auth failed (401) or endpoint rejects auth, try unauthenticated POST
        if code == 401 or code is None:
            try:
                req2 = request.Request(url, data=data_bytes, headers={
                    'Content-Type': 'application/json'
                }, method='POST')
                with request.urlopen(req2, timeout=15) as resp2:
                    body2 = resp2.read().decode('utf-8')
                    return resp2.getcode(), body2, None
            except Exception:
                pass
        return code, body, None
    except Exception as e:
        # Try unauthenticated POST as last resort
        try:
            req2 = request.Request(url, data=data_bytes, headers={
                'Content-Type': 'application/json'
            }, method='POST')
            with request.urlopen(req2, timeout=15) as resp2:
                body2 = resp2.read().decode('utf-8')
                return resp2.getcode(), body2, None
        except Exception as e2:
            return None, str(e2), None


def login_and_post(url, data_bytes, user, pwd):
    """Perform login to /rest/login and reuse session cookie for subsequent POST."""
    login_url = url.rsplit('/', 1)[0] + '/login'
    # n8n login expects an email field (or emailOrLdapLoginId). Send both to be robust.
    login_payload = json.dumps({'email': user, 'password': pwd, 'emailOrLdapLoginId': user}).encode('utf-8')
    login_req = request.Request(login_url, data=login_payload, headers={
        'Content-Type': 'application/json'
    }, method='POST')
    try:
        with request.urlopen(login_req, timeout=15) as resp:
            # Grab Set-Cookie header
            set_cookie = resp.getheader('Set-Cookie')
            if not set_cookie:
                return None, 'Login succeeded but no session cookie returned', None
            # Use returned cookie for the actual request
            req = request.Request(url, data=data_bytes, headers={
                'Content-Type': 'application/json',
                'Cookie': set_cookie
            }, method='POST')
            with request.urlopen(req, timeout=15) as r2:
                body = r2.read().decode('utf-8')
                return r2.getcode(), body, set_cookie
    except error.HTTPError as e:
        try:
            body = e.read().decode('utf-8')
        except Exception:
            body = ''
        return e.code, body, None
    except Exception as e:
        return None, str(e), None


def main():
    wf_path = sys.argv[1] if len(sys.argv)>1 else DEFAULT_JSON
    env = read_env(ENV_PATH)
    user = env.get('N8N_USER')
    pwd = env.get('N8N_PASSWORD')
    if not user or not pwd:
        print('N8N_USER or N8N_PASSWORD not set in .env')
        sys.exit(1)
    try:
        with open(wf_path,'rb') as f:
            wf_json = f.read()
    except Exception as e:
        print('Failed to read workflow JSON:', e)
        sys.exit(1)
    import_url = 'http://localhost:5678/rest/workflows/import'
    code, body, cookie = post_json(import_url, wf_json, user, pwd)
    # If Basic Auth failed (401) try login flow to obtain session cookie
    if code == 401:
        print('Basic Auth failed; attempting login flow...')
        code, body, cookie = login_and_post(import_url, wf_json, user, pwd)
    # If import endpoint not available, try creating workflow at /rest/workflows
    if code == 404:
        print('Import endpoint not found, trying POST to /rest/workflows')
        create_url = 'http://localhost:5678/rest/workflows'
        code, body, cookie = post_json(create_url, wf_json, user, pwd)
        if code == 401:
            print('Basic Auth failed on create; attempting login flow...')
            code, body, cookie = login_and_post(create_url, wf_json, user, pwd)
    print('Import response code:', code)
    print(body)
    if code and (200 <= code < 300):
        try:
            resp = json.loads(body)
            wf_id = resp.get('id')
        except Exception:
            wf_id = None
        if wf_id:
            act_url = f'http://localhost:5678/rest/workflows/{wf_id}/activate'
            # Try Basic Auth first
            code2, body2, cookie2 = post_json(act_url, b'', user, pwd)
            # If 401, try login flow (reuse cookie if available)
            if code2 == 401:
                print('Activation Basic Auth failed; attempting login flow...')
                code2, body2, cookie2 = login_and_post(act_url, b'', user, pwd)
            print('Activate response code:', code2)
            print(body2)
        else:
            print('No workflow id returned from import; cannot activate automatically.')
    else:
        print('Import failed; cannot activate.')

if __name__=='__main__':
    main()
