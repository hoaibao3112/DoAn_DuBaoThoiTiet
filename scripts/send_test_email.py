import os
import smtplib
import ssl
import requests
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
# Recipients CSV: columns -> email,name,city,enabled
RECIPIENTS_CSV = os.getenv('DAILY_RECIPIENTS_CSV', os.path.join('data', 'recipients.csv'))
CITY = os.getenv('DEFAULT_CITY', 'Ho Chi Minh')
# Batch / throttle settings
BATCH_SIZE = int(os.getenv('EMAIL_BATCH_SIZE', '50'))
DELAY_SECONDS = float(os.getenv('EMAIL_DELAY_SECONDS', '1.0'))
LOG_PATH = os.getenv('EMAIL_LOG_PATH', os.path.join('logs', 'email_send.log'))

if not EMAIL_USER or not EMAIL_PASS:
    print('EMAIL_USER or EMAIL_PASS not set in env')
    raise SystemExit(1)

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

import csv
import time
import io
import base64
from datetime import datetime

# Optional plotting (matplotlib). If not available, emails will be sent without charts.
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

def log(msg: str):
    ts = datetime.now().isoformat()
    line = f"[{ts}] {msg}\n"
    with open(LOG_PATH, 'a', encoding='utf-8') as fh:
        fh.write(line)
    print(line, end='')

def build_message(weather_data, to_email, to_name=None, chart_bytes=None):
    """Return a MIME multipart/related message (plain + html + optional inline image)."""
    subject = f"Thông báo thời tiết {weather_data.get('city', CITY)} - {weather_data.get('date', '')}"
    plain = []
    plain.append(f"Thời tiết {weather_data.get('city', CITY)} — {weather_data.get('date', '')}")
    plain.append(f"Nhiệt độ: {weather_data.get('temperature')}°C (Feels like {weather_data.get('feels_like')}°C)")
    plain.append(f"Độ ẩm: {weather_data.get('humidity')}%")
    plain.append(f"Mô tả: {weather_data.get('description')}")
    plain.append(f"AQI: {weather_data.get('aqi')} ({weather_data.get('aqi_category')}) | PM2.5: {weather_data.get('pm25')} | PM10: {weather_data.get('pm10')}")
    plain.append('Khuyến nghị: ' + (weather_data.get('recommendation') or ''))
    plain_body = "\n".join(plain)

    html_body = f"<h3>{weather_data.get('city', CITY)} — {weather_data.get('date','')}</h3>"
    html_body += f"<p><strong>Nhiệt độ:</strong> {weather_data.get('temperature')}°C (Feels like {weather_data.get('feels_like')}°C)</p>"
    html_body += f"<p><strong>Độ ẩm:</strong> {weather_data.get('humidity')}%</p>"
    html_body += f"<p><strong>Mô tả:</strong> {weather_data.get('description')}</p>"
    html_body += f"<p><strong>AQI:</strong> {weather_data.get('aqi')} ({weather_data.get('aqi_category')}) — PM2.5: {weather_data.get('pm25')} | PM10: {weather_data.get('pm10')}</p>"
    html_body += f"<p><strong>Khuyến nghị:</strong> {weather_data.get('recommendation') or ''}</p>"
    unsubscribe_link = f"https://example.com/unsubscribe?email={to_email}"
    html_body += f"<p><a href=\"{unsubscribe_link}\">Hủy nhận thông báo</a></p>"
    if chart_bytes:
        html_body += f"<p><img src=\"cid:chart\" style=\"max-width:600px\"/></p>"

    # Build multipart/related -> multipart/alternative -> (plain, html)
    msg_root = MIMEMultipart('related')
    msg_root['Subject'] = subject
    msg_root['From'] = EMAIL_USER
    msg_root['To'] = to_email

    msg_alternative = MIMEMultipart('alternative')
    msg_root.attach(msg_alternative)

    part_text = MIMEText(plain_body, 'plain', 'utf-8')
    part_html = MIMEText(html_body, 'html', 'utf-8')
    msg_alternative.attach(part_text)
    msg_alternative.attach(part_html)

    if chart_bytes:
        try:
            img = MIMEImage(chart_bytes, 'png')
            img.add_header('Content-ID', '<chart>')
            img.add_header('Content-Disposition', 'inline', filename='chart.png')
            msg_root.attach(img)
            log(f'Attached inline chart (MIMEImage) for {to_email}')
        except Exception as e:
            log(f'Failed to attach chart (MIMEImage) for {to_email}: {e}')

    return msg_root


def make_chart_png_bytes(weather_data) -> bytes:
    """Return PNG bytes showing simple charts for weather + AQI.
    Returns None if plotting unavailable or on error. Logs success/failure."""
    if not HAVE_MPL:
        log('matplotlib not available; skipping chart generation')
        return None
    try:
        temp = float(weather_data.get('temperature') or 0)
        feels = float(weather_data.get('feels_like') or temp)
        hum = float(weather_data.get('humidity') or 0)
        aqi = float(weather_data.get('aqi') or 0)
        pm25 = float(weather_data.get('pm25') or 0)
        pm10 = float(weather_data.get('pm10') or 0)

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
        axes[0].bar(['Temp', 'Feels', 'Humidity'], [temp, feels, hum], color=['#ff7f0e', '#1f77b4', '#2ca02c'])
        axes[0].set_title('Nhiệt độ & Độ ẩm')
        axes[1].bar(['AQI', 'PM2.5', 'PM10'], [aqi, pm25, pm10], color=['#7f7f7f', '#d62728', '#9467bd'])
        axes[1].set_title('AQI & Hạt mịn')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        data = buf.read()
        log('Chart generated successfully')
        return data
    except Exception as e:
        log(f'Chart generation failed: {e}')
        return None


def send_batch(recipients):
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(EMAIL_USER, EMAIL_PASS)

            for r in recipients:
                to = r.get('email')
                name = r.get('name')
                city = r.get('city') or CITY
                try:
                    resp = requests.get(f'http://127.0.0.1:8000/weather/current?city={city}', timeout=10)
                    if resp.status_code != 200:
                        log(f"Failed to fetch weather for {city}: {resp.status_code}")
                        continue
                    data = resp.json()
                except Exception as e:
                    log(f"Error fetching weather for {city}: {e}")
                    continue

                # generate small chart (may return None if matplotlib missing)
                chart_bytes = make_chart_png_bytes(data)
                msg = build_message(data, to, name, chart_bytes=chart_bytes)
                try:
                    server.send_message(msg)
                    log(f"Sent to {to}")
                except Exception as e:
                    log(f"Failed to send to {to}: {e}")
                time.sleep(DELAY_SECONDS)
    except Exception as e:
        log(f"SMTP connection failed: {e}")


def load_recipients(csv_path):
    if not os.path.exists(csv_path):
        log(f"Recipients CSV not found: {csv_path}")
        return []
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            enabled = r.get('enabled','1')
            if str(enabled).strip().lower() in ('0','false','no'):
                continue
            rows.append({'email': r.get('email'), 'name': r.get('name'), 'city': r.get('city')})
    return rows


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def main():
    recipients = load_recipients(RECIPIENTS_CSV)
    if not recipients:
        log('No recipients to send')
        return

    log(f'Starting send to {len(recipients)} recipients (batch_size={BATCH_SIZE}, delay={DELAY_SECONDS}s)')
    for batch in chunk_list(recipients, BATCH_SIZE):
        send_batch(batch)
        log(f'Finished batch of {len(batch)} recipients')


if __name__ == '__main__':
    main()
