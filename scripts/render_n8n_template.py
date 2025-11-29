import json
import re
from html import unescape

# Paths
workflow_path = r"n8n-workflows\Weather_Daily_Email.json"
sample_path = r"tmp_sample.json"
output_path = r"rendered_email.html"

# Load workflow JSON
with open(workflow_path, 'r', encoding='utf-8') as f:
    wf = json.load(f)

# Find the first node that has 'html' in parameters
html_raw = None
for node in wf.get('nodes', []):
    params = node.get('parameters', {})
    if 'html' in params:
        html_raw = params['html']
        break

if html_raw is None:
    print('No html field found in workflow file.')
    raise SystemExit(1)

# The html value may include extra wrapping quotes; strip surrounding double quotes
if isinstance(html_raw, str) and html_raw.startswith('"') and html_raw.endswith('"'):
    html_raw = html_raw[1:-1]

# Unescape JSON-escaped sequences
html_raw = html_raw.encode('utf-8').decode('unicode_escape')
# Also unescape any html entities
html_raw = unescape(html_raw)

# Load sample JSON from backend (saved to tmp_sample.json)
with open(sample_path, 'r', encoding='utf-8-sig') as f:
    sample = json.load(f)

# Helper to replace {{$json["field"]}} and {{$node["GetWeather"].json["field"]}}

def replace_token(match):
    expr = match.group(1).strip()
    # handle forms like $json["city"] or $node["GetWeather"].json["city"]
    # Extract last field name within quotes
    m = re.search(r'\"([^\"]+)\"', expr)
    if not m:
        return ''
    key = m.group(1)
    # prefer sample directly
    val = sample.get(key)
    if val is None:
        # try deeper: sample may be nested under keys
        val = ''
    return str(val)

# pattern for {{$...}}
pattern = re.compile(r'\{\{\s*\$([^\}]+)\s*\}\}')
rendered = pattern.sub(replace_token, html_raw)

# Save output
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(rendered)

print('Rendered HTML saved to', output_path)
print('\n--- Preview ---\n')
print(rendered[:1000])
