"""
CI-friendly email sender for GitHub Actions.
- Reads `data/recipients.csv` for recipients.
- Uses OpenWeather Geocoding + Current Weather + Air Pollution APIs to build content.
- Generates a small PNG chart (matplotlib) and embeds it as inline image.
- Sends email via SMTP using `EMAIL_USER` and `EMAIL_PASS` secrets.

Set GitHub secrets: EMAIL_USER, EMAIL_PASS, OPENWEATHER_API_KEY
"""
import os
import csv
import io
import smtplib
import ssl
import requests
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

OPENWEATHER_KEY = os.getenv('OPENWEATHER_API_KEY')
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
RECIPIENTS_CSV = os.getenv('DAILY_RECIPIENTS_CSV', os.path.join('data', 'recipients.csv'))

if not OPENWEATHER_KEY:
    print('OPENWEATHER_API_KEY not set. Exiting.')
    raise SystemExit(1)
if not EMAIL_USER or not EMAIL_PASS:
    print('EMAIL_USER or EMAIL_PASS not set. Exiting.')
    raise SystemExit(1)

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def log(msg):
    print(f"[{datetime.utcnow().isoformat()}] {msg}")


def geocode_city(city):
    url = 'http://api.openweathermap.org/geo/1.0/direct'
    params = {'q': city, 'limit': 1, 'appid': OPENWEATHER_KEY}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        log(f'Geocode failed for {city}: {r.status_code}')
        return None
    js = r.json()
    if not js:
        log(f'No geocode result for {city}')
        return None
    return js[0].get('lat'), js[0].get('lon')


def fetch_weather(lat, lon):
    wurl = 'https://api.openweathermap.org/data/2.5/weather'
    params = {'lat': lat, 'lon': lon, 'units': 'metric', 'appid': OPENWEATHER_KEY}
    r = requests.get(wurl, params=params, timeout=10)
    if r.status_code != 200:
        log(f'Weather fetch failed: {r.status_code}')
        return None
    return r.json()


def fetch_air(lat, lon):
    aurl = 'http://api.openweathermap.org/data/2.5/air_pollution'
    params = {'lat': lat, 'lon': lon, 'appid': OPENWEATHER_KEY}
    r = requests.get(aurl, params=params, timeout=10)
    if r.status_code != 200:
        log(f'Air fetch failed: {r.status_code}')
        return None
    return r.json()


def make_chart_bytes(temp, feels, hum, aqi, pm25, pm10):
    if not HAVE_MPL:
        return None
    try:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
        axes[0].bar(['Temp', 'Feels', 'Humidity'], [temp, feels, hum], color=['#ff7f0e', '#1f77b4', '#2ca02c'])
        axes[0].set_title('Nhiệt độ & Độ ẩm')
        axes[1].bar(['AQI', 'PM2.5', 'PM10'], [aqi, pm25, pm10], color=['#7f7f7f', '#d62728', '#9467bd'])
        axes[1].set_title('AQI & Hạt mịn')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        log(f'Chart generation error: {e}')
        return None


def build_message(city_label, weather, air, chart_bytes, to_email):
    temp = weather.get('main', {}).get('temp')
    feels = weather.get('main', {}).get('feels_like')
    hum = weather.get('main', {}).get('humidity')
    description = (weather.get('weather') or [{}])[0].get('description')

    aqi = None
    pm25 = None
    pm10 = None
    if air and 'list' in air and air['list']:
        comp = air['list'][0].get('components', {})
        pm25 = comp.get('pm2_5')
        pm10 = comp.get('pm10')
        aqi = air['list'][0].get('main', {}).get('aqi')

    subject = f"Thông báo thời tiết {city_label} - {datetime.utcnow().date()}"

    plain = []
    plain.append(f"Thời tiết {city_label} - {datetime.utcnow().date()}")
    plain.append(f"Nhiệt độ: {temp}°C (Feels like {feels}°C)")
    plain.append(f"Độ ẩm: {hum}%")
    plain.append(f"Mô tả: {description}")
    plain.append(f"AQI: {aqi} | PM2.5: {pm25} | PM10: {pm10}")
    plain_body = "\n".join(plain)

    html = f"<h3>{city_label} — {datetime.utcnow().date()}</h3>"
    html += f"<p><strong>Nhiệt độ:</strong> {temp}°C (Feels like {feels}°C)</p>"
    html += f"<p><strong>Độ ẩm:</strong> {hum}%</p>"
    html += f"<p><strong>Mô tả:</strong> {description}</p>"
    html += f"<p><strong>AQI:</strong> {aqi} — PM2.5: {pm25} | PM10: {pm10}</p>"
    html += "<p><a href=\"https://example.com/unsubscribe\">Hủy nhận</a></p>"
    if chart_bytes:
        html += f"<p><img src=\"cid:chart\" style=\"max-width:600px\"/></p>"

    msg_root = MIMEMultipart('related')
    msg_root['Subject'] = subject
    msg_root['From'] = EMAIL_USER
    msg_root['To'] = to_email

    msg_alt = MIMEMultipart('alternative')
    msg_root.attach(msg_alt)
    msg_alt.attach(MIMEText(plain_body, 'plain', 'utf-8'))
    msg_alt.attach(MIMEText(html, 'html', 'utf-8'))

    if chart_bytes:
        try:
            img = MIMEImage(chart_bytes, 'png')
            img.add_header('Content-ID', '<chart>')
            img.add_header('Content-Disposition', 'inline', filename='chart.png')
            msg_root.attach(img)
        except Exception as e:
            log(f'Attach image error: {e}')

    return msg_root


def load_recipients(path):
    if not os.path.exists(path):
        log(f'Recipients file not found: {path}')
        return []
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if str(r.get('enabled','1')).strip().lower() in ('0','false','no'):
                continue
            rows.append(r)
    return rows


def main():
    recipients = load_recipients(RECIPIENTS_CSV)
    if not recipients:
        log('No recipients. Exiting.')
        return

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(EMAIL_USER, EMAIL_PASS)

        for r in recipients:
            email = r.get('email')
            city = r.get('city') or 'Ho Chi Minh'
            city_label = r.get('name') or city
            try:
                geo = geocode_city(city)
                if not geo:
                    log(f'Skipping {email}: geocode failed')
                    continue
                lat, lon = geo
                weather = fetch_weather(lat, lon)
                air = fetch_air(lat, lon)
                temp = weather.get('main',{}).get('temp',0)
                feels = weather.get('main',{}).get('feels_like',temp)
                hum = weather.get('main',{}).get('humidity',0)
                # find aqi and particulates
                aqi = 0
                pm25 = 0
                pm10 = 0
                if air and 'list' in air and air['list']:
                    comp = air['list'][0].get('components',{})
                    pm25 = comp.get('pm2_5',0)
                    pm10 = comp.get('pm10',0)
                    aqi = air['list'][0].get('main',{}).get('aqi',0)

                chart = make_chart_bytes(temp, feels, hum, aqi, pm25, pm10)
                msg = build_message(city_label, weather, air, chart, email)
                server.send_message(msg)
                log(f'Sent to {email}')
            except Exception as e:
                log(f'Error for {email}: {e}')

if __name__ == '__main__':
    main()
