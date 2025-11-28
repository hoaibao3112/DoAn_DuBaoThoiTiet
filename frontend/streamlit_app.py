import streamlit as st
import requests
import unicodedata
import os
import time
import json
import re
import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dateparser
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import random
import html

# Folium may not be installed in every runtime (e.g., inside Docker image). Import safely.
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    folium = None
    st_folium = None
    FOLIUM_AVAILABLE = False


def _aqi_color(val):
    try:
        v = float(val)
    except Exception:
        return '#999999'
    if v <= 50:
        return '#00c853'  # green
    if v <= 100:
        return '#ffeb3b'  # yellow
    if v <= 150:
        return '#ff9100'  # orange
    if v <= 200:
        return '#f44336'  # red
    if v <= 300:
        return '#9c27b0'  # purple
    return '#6d0000'      # maroon


def fix_text_encoding(s: str) -> str:
    """Attempt to fix common mojibake where UTF-8 bytes were decoded as latin-1.
    If the string appears to contain mojibake artifacts (Ã, â, Ä etc.), try
    re-encoding as latin-1 and decoding as utf-8. Otherwise return original.
    """
    if not s or not isinstance(s, str):
        return s
    # quick heuristic: presence of common mojibake characters
    mojibake_indicators = ['Ã', 'â', 'Ä', '\ufffd']
    try:
        if any(ch in s for ch in mojibake_indicators):
            fixed = s.encode('latin-1')
            try:
                return fixed.decode('utf-8')
            except Exception:
                # if decode fails, return original
                return s
    except Exception:
        return s
    return s


def make_popup_html(data: dict) -> str:
    """Return simple HTML for folium popup given cleaned data dict."""
    try:
        city = html.escape(str(data.get('city', '')))
        temp = data.get('weather', {}).get('temp') or data.get('temp') or 'N/A'
        desc = html.escape(str(data.get('weather', {}).get('description') or ''))
        aqi = data.get('analysis', {}).get('aqi') or data.get('aqi') or 'N/A'
        pm25 = data.get('aqi', {}).get('pm25') if isinstance(data.get('aqi'), dict) else data.get('pm25')
        pm25 = pm25 or 'N/A'
        html_parts = [f"<b>{city}</b>", f"Mô tả: {desc}", f"Nhiệt độ: {temp}°C", f"AQI: {aqi}", f"PM2.5: {pm25}"]
        return '<br>'.join(html_parts)
    except Exception:
        return 'Dữ liệu không đầy đủ'


def show_map(lat: float, lon: float, data: dict = None, zoom: int = 12, width: int = 700, height: int = 450):
    """Render a Folium map in Streamlit centered at (lat, lon) with an optional popup showing `data`."""
    if not FOLIUM_AVAILABLE:
        st.error('Thư viện `folium` hoặc `streamlit_folium` chưa được cài trong môi trường này. Nếu bạn đang dùng Docker, hãy rebuild image streamlit để cài các dependencies (xem hướng dẫn trong terminal).')
        return
    m = folium.Map(location=[lat, lon], zoom_start=zoom)
    popup_html = make_popup_html(data or {})
    # determine color from AQI if available
    aqi_val = None
    try:
        aqi_val = data.get('analysis', {}).get('aqi') if data else None
    except Exception:
        aqi_val = None

    def _color_from_aqi(v):
        try:
            vv = float(v)
        except Exception:
            return '#888888'
        if vv <= 50:
            return '#00c853'
        if vv <= 100:
            return '#ffeb3b'
        if vv <= 150:
            return '#ff9100'
        if vv <= 200:
            return '#f44336'
        if vv <= 300:
            return '#9c27b0'
        return '#6d0000'

    color = _color_from_aqi(aqi_val)
    folium.CircleMarker(location=[lat, lon], radius=10, color=color, fill=True, fill_opacity=0.8, popup=popup_html).add_to(m)
    # display map
    st_folium(m, width=width, height=height)


def extract_coords_from_cleaned(cleaned: dict):
    """Try multiple possible locations in `cleaned` payload to find coordinates.
    Returns dict with 'lat' and 'lon' or None.
    """
    if not cleaned or not isinstance(cleaned, dict):
        return None
    # common places
    try:
        # weather.coord structure
        w = cleaned.get('weather')
        if isinstance(w, dict):
            c = w.get('coord')
            if isinstance(c, dict) and ('lat' in c or 'latitude' in c):
                # normalize keys
                lat = c.get('lat') or c.get('latitude')
                lon = c.get('lon') or c.get('longitude')
                if lat is not None and lon is not None:
                    return {'lat': lat, 'lon': lon}
        # top-level coord
        c = cleaned.get('coord')
        if isinstance(c, dict) and ('lat' in c or 'latitude' in c):
            lat = c.get('lat') or c.get('latitude')
            lon = c.get('lon') or c.get('longitude')
            if lat is not None and lon is not None:
                return {'lat': lat, 'lon': lon}
        # raw payload (OpenWeather style)
        # Note: weather raw may be nested under cleaned['weather']['raw']
        w = cleaned.get('weather') or {}
        if isinstance(w, dict):
            wr = w.get('raw') or {}
            if isinstance(wr, dict) and ('lat' in wr or 'lon' in wr or 'coord' in wr):
                c = wr.get('coord') or wr
                if isinstance(c, dict) and ('lat' in c or 'latitude' in c):
                    return (c.get('lat'), c.get('lon') or c.get('longitude'))

        raw = cleaned.get('raw') or {}
        if isinstance(raw, dict):
            c = raw.get('coord')
            if isinstance(c, dict) and ('lat' in c or 'latitude' in c):
                lat = c.get('lat') or c.get('latitude')
                lon = c.get('lon') or c.get('longitude')
                if lat is not None and lon is not None:
                    return {'lat': lat, 'lon': lon}
            # sometimes raw includes lat/lon at root
            if 'lat' in raw and 'lon' in raw:
                return {'lat': raw.get('lat'), 'lon': raw.get('lon')}
        # some services return geometry->location
        geom = cleaned.get('geometry') or (raw.get('geometry') if isinstance(raw, dict) else None)
        if isinstance(geom, dict):
            loc = geom.get('location')
            if isinstance(loc, dict) and 'lat' in loc and 'lng' in loc:
                return {'lat': loc.get('lat'), 'lon': loc.get('lng')}
    except Exception:
        return None
    return None


def categorize_pollutant(pollutant: str, value) -> tuple:
    """Return (category, short_advice) for common pollutants using simple breakpoints.
    pollutant: name like 'pm25','pm10','o3','no2','so2','co'
    value: numeric concentration (µg/m3 for PM/O3/NO2/SO2, mg/m3 or µg/m3 for CO depending)
    """
    try:
        v = float(value)
    except Exception:
        return (None, '')

    p = pollutant.lower()
    # Simple categories (approximate, for quick advice)
    if p in ('pm25', 'pm_25', 'pm2.5', 'pm2_5', 'pm25'):
        if v <= 12:
            return ('Good', 'Không khí tốt cho mọi người.')
        if v <= 35.4:
            return ('Moderate', 'Người nhạy cảm nên chú ý.')
        if v <= 55.4:
            return ('Unhealthy for Sensitive', 'Người bệnh hô hấp nên hạn chế hoạt động ngoài trời.')
        if v <= 150.4:
            return ('Unhealthy', 'Hạn chế ra ngoài, đeo khẩu trang phù hợp.')
        if v <= 250.4:
            return ('Very Unhealthy', 'Tránh ra ngoài; cân nhắc ở nhà.')
        return ('Hazardous', 'Nguy hiểm: hạn chế tối đa tiếp xúc ngoài trời.')

    if p in ('pm10', 'pm_10'):
        if v <= 54:
            return ('Good', 'Không khí tốt cho mọi người.')
        if v <= 154:
            return ('Moderate', 'Người nhạy cảm nên chú ý.')
        if v <= 254:
            return ('Unhealthy for Sensitive', 'Người bệnh hô hấp nên hạn chế hoạt động ngoài trời.')
        if v <= 354:
            return ('Unhealthy', 'Hạn chế ra ngoài, đeo khẩu trang phù hợp.')
        if v <= 424:
            return ('Very Unhealthy', 'Tránh ra ngoài; cân nhắc ở nhà.')
        return ('Hazardous', 'Nguy hiểm: hạn chế tối đa tiếp xúc ngoài trời.')

    if p in ('o3', 'ozone'):
        if v <= 54:
            return ('Good', 'Không khí tốt.')
        if v <= 70:
            return ('Moderate', 'Người nhạy cảm nên chú ý.')
        if v <= 85:
            return ('Unhealthy for Sensitive', 'Người bệnh nên hạn chế.')
        if v <= 105:
            return ('Unhealthy', 'Hạn chế hoạt động ngoài trời.')
        if v <= 200:
            return ('Very Unhealthy', 'Tránh ra ngoài nếu có thể.')
        return ('Hazardous', 'Nguy hiểm.')

    if p in ('no2', 'no_2'):
        if v <= 53:
            return ('Good', 'Không khí tốt.')
        if v <= 100:
            return ('Moderate', 'Người nhạy cảm nên chú ý.')
        if v <= 360:
            return ('Unhealthy for Sensitive', 'Người bệnh hô hấp nên hạn chế.')
        if v <= 649:
            return ('Unhealthy', 'Hạn chế ra ngoài.')
        return ('Very Unhealthy', 'Nguy hiểm.')

    if p in ('so2', 'so_2'):
        if v <= 35:
            return ('Good', 'Không khí tốt.')
        if v <= 75:
            return ('Moderate', 'Người nhạy cảm nên chú ý.')
        if v <= 185:
            return ('Unhealthy for Sensitive', 'Người bệnh hô hấp nên hạn chế.')
        if v <= 304:
            return ('Unhealthy', 'Hạn chế ra ngoài.')
        return ('Very Unhealthy', 'Nguy hiểm.')

    if p in ('co',):
        # CO typically in mg/m3 or µg/m3; use simple µg/m3 thresholds if value small assume mg/m3
        if v <= 4.4:
            return ('Good', 'Không khí tốt.')
        if v <= 9.4:
            return ('Moderate', 'Người nhạy cảm nên chú ý.')
        if v <= 12.4:
            return ('Unhealthy for Sensitive', 'Người bệnh nên hạn chế.')
        if v <= 15.4:
            return ('Unhealthy', 'Hạn chế ra ngoài.')
        return ('Very Unhealthy', 'Nguy hiểm.')

    return (None, '')


def find_statistical_neighbors(csv_path, target_station, metric_tokens=('pm25','pm10'), min_overlap=120, top_k=3):
    """Find neighbor stations statistically similar to `target_station`.
    Approach: pivot time series per station for a chosen metric (tries tokens in order),
    compute Pearson correlation on overlapping dates, require at least `min_overlap` days,
    and return top_k neighbors sorted by correlation.
    Returns list of tuples: (station_id, corr, overlap_days).
    """
    try:
        df_all = pd.read_csv(csv_path, low_memory=False)
        if 'StationId' not in df_all.columns and 'stationid' in [c.lower() for c in df_all.columns]:
            # try case-insensitive match
            for c in df_all.columns:
                if c.lower() == 'stationid':
                    df_all.rename(columns={c: 'StationId'}, inplace=True)
                    break
        if 'StationId' not in df_all.columns or 'Date' not in df_all.columns:
            return []

        df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
        df_all = df_all.dropna(subset=['Date'])

        # choose a metric column present in the CSV using tokens
        norm_map = {re.sub(r'[^0-9a-z]', '', c.lower()): c for c in df_all.columns}
        metric_col = None
        for tok in metric_tokens:
            if tok in norm_map:
                metric_col = norm_map[tok]
                break
        # fallback: try to find any column that contains the token
        if not metric_col:
            for n, orig in norm_map.items():
                for tok in metric_tokens:
                    if tok in n:
                        metric_col = orig
                        break
                if metric_col:
                    break

        if not metric_col:
            return []

        # pivot to station x date
        df_pivot = df_all[['StationId','Date',metric_col]].copy()
        df_pivot[metric_col] = pd.to_numeric(df_pivot[metric_col], errors='coerce')
        df_pivot = df_pivot.dropna(subset=[metric_col])
        if df_pivot.empty:
            return []

        pivot = df_pivot.pivot_table(index='Date', columns='StationId', values=metric_col)

        if target_station not in pivot.columns:
            return []

        target_series = pivot[target_station]
        neighbors = []
        for col in pivot.columns:
            if col == target_station:
                continue
            s = pivot[col]
            both = target_series.notna() & s.notna()
            overlap = int(both.sum())
            if overlap < min_overlap:
                continue
            try:
                corr = float(target_series[both].corr(s[both]))
            except Exception:
                corr = 0.0
            neighbors.append((col, corr if not pd.isna(corr) else 0.0, overlap))

        # sort by absolute correlation descending
        neighbors = sorted(neighbors, key=lambda x: abs(x[1]), reverse=True)
        return neighbors[:top_k]
    except Exception:
        return []


def weighted_neighbor_rain_prob(csv_path, target_station, neighbors, precip_threshold=0.5, precip_tokens=('rain','precip')):
    """Estimate rain probability for target station using neighbors and their weights.
    neighbors: list of (station_id, corr, overlap)
    Returns weighted_prob (0-100) and raw details list.
    """
    try:
        if not neighbors:
            return None, []
        df_all = pd.read_csv(csv_path, low_memory=False)
        df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
        df_all = df_all.dropna(subset=['Date'])

        # pick precipitation column heuristically
        precip_col = None
        for c in df_all.columns:
            lc = c.lower()
            if any(tok in lc for tok in precip_tokens):
                precip_col = c
                break

        details = []
        weights = []
        probs = []
        for sid, corr, overlap in neighbors:
            df_s = df_all[df_all['StationId'] == sid]
            if precip_col and precip_col in df_s.columns:
                p = pd.to_numeric(df_s[precip_col], errors='coerce').dropna()
                if len(p) == 0:
                    prob = None
                else:
                    prob = 100.0 * (p > precip_threshold).sum() / len(p)
            else:
                # fallback: try description
                desc_cols = [c for c in df_s.columns if any(x in c.lower() for x in ('desc','weather','note'))]
                if desc_cols:
                    dc = df_s[desc_cols[0]].astype(str).fillna('').str.lower()
                    prob = 100.0 * dc.str.contains('mua|mưa|rain').sum() / len(dc) if len(dc) > 0 else None
                else:
                    prob = None
            weight = abs(corr) if corr is not None else 0.0
            weights.append(weight)
            probs.append(prob if prob is not None else 0.0)
            details.append({'station': sid, 'corr': corr, 'overlap': overlap, 'prob': prob})

        total_w = sum(weights) if sum(weights) > 0 else 0.0
        if total_w <= 0:
            return None, details
        weighted = sum(p * w for p, w in zip(probs, weights)) / total_w
        return float(weighted), details
    except Exception:
        return None, []


def detect_future_query(text: str):
    """Detect whether user asked about a future date or range.
    Returns tuple (is_future: bool, key: str or None, params: dict or None)
    Supported keys:
      - 'tomorrow' (no params)
      - 'date' with params {'date': Timestamp}
      - 'range' with params {'start': Timestamp, 'end': Timestamp}
    Recognizes Vietnamese phrases like 'ngày mai', explicit dates '25/11' or '25-11-2025', and
    ranges like 'tuần tới' / 'tuần sau'.
    """
    import re
    from dateutil import parser
    from datetime import datetime

    if not text or not isinstance(text, str):
        return (False, None, None)
    t = text.lower()
    # tomorrow
    if 'ngày mai' in t or re.search(r'\bmai\b', t):
        return (True, 'tomorrow', None)

    # Try using dateparser for flexible natural language dates first
    try:
        dp = dateparser.parse(t, languages=['vi'], settings={'PREFER_DATES_FROM': 'future'})
        if dp is not None:
            # if dateparser returns a datetime in the future, accept it as explicit date
            now = pd.Timestamp.utcnow()
            dt = pd.Timestamp(dp)
            if dt.normalize() >= now.normalize():
                return (True, 'date', {'date': dt})
    except Exception:
        pass

    # explicit date patterns: dd/mm[/yyyy] or dd-mm[-yyyy]
    m = re.search(r"(?:(?:ngày)\s*)?(\d{1,2})[\/-](\d{1,2})(?:[\/-](\d{2,4}))?", t)
    if m:
        day = int(m.group(1))
        month = int(m.group(2))
        year = m.group(3)
        try:
            if year:
                y = int(year)
                if y < 100:  # assume 20xx for two-digit years
                    y += 2000
                dt = pd.Timestamp(year=y, month=month, day=day)
            else:
                # assume next occurrence (this year or next if already past)
                now = pd.Timestamp.utcnow()
                try:
                    dt = pd.Timestamp(year=now.year, month=month, day=day)
                    if dt.normalize() <= now.normalize():
                        # pick next year
                        dt = pd.Timestamp(year=now.year + 1, month=month, day=day)
                except Exception:
                    dt = pd.Timestamp.utcnow()  # fallback
            return (True, 'date', {'date': dt})
        except Exception:
            pass

    # week ranges: 'tuần tới', 'tuần sau', 'tuần này'
    if 'tuần tới' in t or 'tuần sau' in t:
        today = pd.Timestamp.utcnow().normalize()
        # compute next week's Monday
        weekday = today.weekday()  # Monday=0
        days_to_next_monday = (7 - weekday) % 7
        if days_to_next_monday == 0:
            days_to_next_monday = 7
        start = today + pd.Timedelta(days=days_to_next_monday)
        end = start + pd.Timedelta(days=6)
        return (True, 'range', {'start': start, 'end': end})

    if 'tuần này' in t:
        today = pd.Timestamp.utcnow().normalize()
        weekday = today.weekday()
        start = today - pd.Timedelta(days=weekday)
        end = start + pd.Timedelta(days=6)
        return (True, 'range', {'start': start, 'end': end})

    return (False, None, None)


def detect_weather_intent(text: str) -> bool:
    """Stricter weather intent detection at module level.
    - Require an explicit weather keyword (thời tiết, mưa, nắng, gió, nhiệt độ, AQI, ô nhiễm)
    - OR a city mention (HCM, Hà Nội, Saigon, Hanoi)
    Returns True when the text likely queries the weather.
    """
    def _norm(s: str) -> str:
        if not s:
            return ""
        return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch)).lower()

    text_norm = _norm(text)
    weather_keywords = ["thoi tiet", "thời tiết", "nhiet do", "nhiệt độ", "mưa", "nắng", "gió", "gio", "aqi", "o nhiem", "ô nhiễm", "ô-nhiễm", "pm2", "pm2.5", "pm10"]
    city_tokens = ["hcm", "ho chi minh", "hochiminh", "saigon", "sai gon", "hanoi", "ha noi"]

    has_weather_kw = any(k in text_norm for k in weather_keywords)
    has_city = any(c in text_norm for c in city_tokens)

    if has_weather_kw:
        return True

    date_indicators = ['mai', 'ngay', 'ngày', 'tuần', 'tuan', 'hôm', 'hom']
    has_date = any(d in text_norm for d in date_indicators)
    if has_city and has_date:
        return True
    return False


def extract_city(text: str) -> str:
    """Extract a city name from text using simple heuristics; returns a readable city string."""
    def _norm(s: str) -> str:
        if not s:
            return ""
        return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch)).lower()

    text_norm = _norm(text)
    if any(tok in text_norm for tok in ["hcm", "ho chi minh", "hochiminh", "thanh pho ho chi minh", "saigon", "sai gon"]):
        return "Ho Chi Minh"
    if any(tok in text_norm for tok in ["hanoi", "ha noi"]):
        return "Hanoi"
    return os.getenv("DEFAULT_CITY", "Hanoi")


def detect_query_type(text: str) -> str:
    """Classify user query into a small set of types used by the UI.
    Returns one of: 'now', 'forecast', 'trend', 'compare', 'pollution', 'map', 'unknown'
    """
    if not text or not isinstance(text, str):
        return 'unknown'
    t = text.lower()
    if any(k in t for k in ('dự đoán', 'dự báo', 'dự đoán', 'forecast', 'ngày mai', 'mai')):
        return 'forecast'
    if any(k in t for k in ('xu hướng', 'xu hướng', 'trend', 'tăng', 'giảm', 'tăng lên', 'giảm đi')):
        return 'trend'
    if any(k in t for k in ('so sánh', 'so sánh', 'vs', 'v.s.', 'và')) and ('so sánh' in t or ' vs ' in t or ' và ' in t):
        return 'compare'
    if any(k in t for k in ('khuyến nghị', 'khuyến cáo', 'nên', 'gợi ý', 'lời khuyên', 'advice')):
        return 'pollution'
    if any(k in t for k in ('bản đồ', 'map', 'tọa độ', 'tọa độ')):
        return 'map'
    # default to 'now' for weather-like inputs
    return 'now'


def _normalize_city_token(tok: str) -> str:
    tokn = ''.join(ch for ch in unicodedata.normalize('NFKD', tok) if not unicodedata.combining(ch)).lower()
    if any(x in tokn for x in ('hcm', 'ho chi minh', 'hochiminh', 'saigon')):
        return 'Ho Chi Minh'
    if any(x in tokn for x in ('hanoi', 'ha noi', 'hn')):
        return 'Hanoi'
    return tok.strip().title()


def extract_cities_for_compare(text: str) -> list:
    """Try to extract two city names from a comparison query using simple separators."""
    t = text.lower()
    # try Vietnamese 'và' or 'vs'
    if ' so sánh ' in t:
        parts = re.split(r'so sánh|vs\.?|v\.?s\.?| và ', t)
    else:
        parts = re.split(r'vs\.?|v\.?s\.?| và |,', t)
    # Keep tokens that look like city names (simple heuristic)
    cities = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # take first 3 words as potential city token
        tok = ' '.join(p.split()[:3])
        norm = _normalize_city_token(tok)
        cities.append(norm)
        if len(cities) >= 2:
            break
    return cities


def sarima_forecast(series: pd.Series, steps: int = 1):
    """Fit a lightweight SARIMA and forecast `steps` ahead.
    Returns (forecast_values, lower_bounds, upper_bounds) as numpy arrays.
    On failure, raises or returns None.
    """
    import numpy as np

    # series: pd.Series indexed by datetime (or monotonic date-like index)
    if series is None or len(series.dropna()) < 10:
        raise ValueError('Insufficient data for SARIMA')

    # ensure datetime index
    s = series.dropna().copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        # try to coerce index
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            s.index = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(s))

    # basic SARIMAX specification with weekly seasonality
    order = (1, 1, 1)
    seasonal_order = (1, 0, 1, 7)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    try:
        model = SARIMAX(s, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=steps)
        mean = pred.predicted_mean.values
        ci = pred.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
        return mean, lower, upper
    except Exception as e:
        raise



def forecast_from_csv(request_key: str, city: str = None, csv_path: str = None, params: dict = None) -> str:
    """Return a short Vietnamese forecast string derived from historical CSV data.
    request_key: 'tomorrow' for now.
    city: optional city filter.
    csv_path: optional path to CSV; defaults to repo `data/cleaned/station_day_clean.csv`.
    """
    from pathlib import Path
    try:
        if csv_path is None:
            # assume repo root contains `data/cleaned/station_day_clean.csv`
            csv_path = Path.cwd() / 'data' / 'cleaned' / 'station_day_clean.csv'
        else:
            csv_path = Path(csv_path)
        if not csv_path.exists():
            return 'Không có dữ liệu lịch sử (file CSV không tìm thấy) nên không thể dự đoán.'

        df = pd.read_csv(csv_path, low_memory=False)
        # Try to find a date column
        date_col = None
        for c in ['Date', 'date', 'timestamp', 'timestamp_utc']:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            # try first column
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

        if city and 'city' in df.columns:
            df_city = df[df['city'].astype(str).str.lower() == str(city).lower()]
        else:
            df_city = df

        if df_city.empty:
            return 'Không tìm thấy bản ghi lịch sử cho thành phố này; không thể dự đoán chính xác.'

        # For 'tomorrow', prefer using the most recent 30 days as the training window
        if request_key == 'tomorrow':
            target_date = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
            time_desc = f"ngày {target_date.day}/{target_date.month}"
            # Use most recent 30 calendar days from df_city
            try:
                df_city_sorted = df_city.sort_values(by=date_col)
                # Take last 30 unique days
                df_city_sorted[date_col] = pd.to_datetime(df_city_sorted[date_col])
                last_day = df_city_sorted[date_col].max().normalize()
                window_start = last_day - pd.Timedelta(days=29)
                cand = df_city_sorted[(df_city_sorted[date_col] >= window_start) & (df_city_sorted[date_col] <= last_day)]
            except Exception:
                # fallback to same-month-day matching if any error
                month = target_date.month
                day = target_date.day
                cand = df_city[(df_city[date_col].dt.month == month) & (df_city[date_col].dt.day == day)]
        elif request_key == 'date' and params and params.get('date') is not None:
            tgt = pd.Timestamp(params.get('date')).normalize()
            month = tgt.month
            day = tgt.day
            cand = df_city[(df_city[date_col].dt.month == month) & (df_city[date_col].dt.day == day)]
            time_desc = f"ngày {tgt.day}/{tgt.month}/{tgt.year}"
        elif request_key == 'range' and params and params.get('start') and params.get('end'):
            start = pd.Timestamp(params.get('start')).normalize()
            end = pd.Timestamp(params.get('end')).normalize()
            # build list of (month,day) in range (handle year wrap)
            rng = []
            cur = start
            while cur <= end:
                rng.append((cur.month, cur.day))
                cur = cur + pd.Timedelta(days=1)
            cand = df_city[df_city[date_col].dt.to_period('D').apply(lambda d: (int(d.month), int(d.day))).isin(rng)]
            time_desc = f"từ {start.date()} đến {end.date()}"
        else:
            return 'Hiện chỉ hỗ trợ dự đoán cho "ngày mai", ngày cụ thể (dd/mm) và các tuần (tuần tới, tuần này).'

        if cand is None or len(cand) == 0:
            # fallback to same month
            cand = df_city[df_city[date_col].dt.month == month]
            fallback = True
        else:
            fallback = False

        if len(cand) == 0:
            return f'Không có dữ liệu lịch sử đủ cho {time_desc}; không thể đưa ra dự đoán.'

        # Choose columns if exist (robust to common column-name variants like 'PM2.5', 'PM10', 'AQI')
        metrics = {}
        try:
            # build normalized map of columns: remove non-alphanumeric and lowercase
            norm_to_col = {}
            for c in cand.columns:
                norm = re.sub(r'[^0-9a-z]', '', c.lower())
                if norm and norm not in norm_to_col:
                    norm_to_col[norm] = c

            # expose detected columns for quick UI debugging
            try:
                st.session_state['detected_csv_columns'] = dict(norm_to_col)
            except Exception:
                pass

            # Attempt simple imputation from lag/ma columns when primary metric columns missing
            # e.g., PM2.5_lag1, PM2.5_ma3, pm25_ma7 -> normalized keys like 'pm25lag1','pm25ma3'
            imputation_candidates = {}
            for norm, orig in norm_to_col.items():
                # detect patterns like pm25lag, pm25ma, pm25std
                for m in ('pm25', 'pm10', 'temp', 'aqi'):
                    if norm.startswith(m) and any(tok in norm for tok in ('lag', 'ma', 'std')):
                        imputation_candidates.setdefault(m, []).append(orig)

            def _find_col(preferred_norms):
                for n in preferred_norms:
                    if n in norm_to_col:
                        return norm_to_col[n]
                return None

            targets = {
                'temp': ['temp', 'temperature'],
                'pm25': ['pm25'],
                'pm10': ['pm10'],
                'aqi': ['aqi']
            }
            for key, prefs in targets.items():
                colname = _find_col(prefs)
                if colname:
                    s = pd.to_numeric(cand[colname], errors='coerce').dropna()
                    if len(s) > 0:
                        metrics[key] = (s.mean(), s.std(), len(s))
            # if metrics still missing, try imputation using available lag/ma columns
            for m in ('pm25', 'pm10', 'temp', 'aqi'):
                if m not in metrics and m in imputation_candidates:
                    vals = []
                    for col in imputation_candidates.get(m, []):
                        try:
                            s = pd.to_numeric(cand[col], errors='coerce').dropna()
                            if len(s) > 0:
                                vals.append(s.mean())
                        except Exception:
                            continue
                    if vals:
                        # use mean of imputation sources as estimate
                        est_mean = float(pd.Series(vals).mean())
                        est_sd = float(pd.Series(vals).std()) if len(vals) > 1 else 0.0
                        metrics[m] = (est_mean, est_sd, sum([int(pd.to_numeric(cand[col], errors='coerce').dropna().shape[0]) for col in imputation_candidates.get(m, [])]))
        except Exception:
            metrics = {}

        if not metrics:
            # Climatology fallback: compute day-of-year (preferred) or monthly means
            try:
                cand[date_col] = pd.to_datetime(cand[date_col], errors='coerce')
            except Exception:
                pass

            # determine target date for matching (tomorrow / explicit date / start of range)
            try:
                if request_key == 'tomorrow':
                    tgt = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
                elif request_key == 'date' and params and params.get('date') is not None:
                    tgt = pd.Timestamp(params.get('date')).normalize()
                elif request_key == 'range' and params and params.get('start'):
                    tgt = pd.Timestamp(params.get('start')).normalize()
                else:
                    tgt = None
            except Exception:
                tgt = None

            # Build normalized column map for candidate selection
            norm_to_col = {}
            for c in cand.columns:
                try:
                    norm = re.sub(r'[^0-9a-z]', '', c.lower())
                except Exception:
                    norm = c.lower()
                if norm and norm not in norm_to_col:
                    norm_to_col[norm] = c
            # expose detected columns for quick UI debugging
            try:
                st.session_state['detected_csv_columns'] = dict(norm_to_col)
            except Exception:
                pass

            def _pick_col(preferred):
                for p in preferred:
                    if p in norm_to_col:
                        return norm_to_col[p]
                # fallback: find column whose norm contains token
                for n, orig in norm_to_col.items():
                    for p in preferred:
                        if p in n:
                            return orig
                return None

            targets = {
                'pm25': ['pm25'],
                'pm10': ['pm10'],
                'temp': ['temp', 'temperature'],
                'aqi': ['aqi']
            }

            # select matching rows for climatology: same month/day if possible
            if tgt is not None and date_col in cand.columns and pd.api.types.is_datetime64_any_dtype(cand[date_col]):
                same = cand[(cand[date_col].dt.month == tgt.month) & (cand[date_col].dt.day == tgt.day)]
                fallback_note = f"(dựa trên cùng ngày qua các năm)"
            else:
                # use same month if tgt available, otherwise use whole cand
                if tgt is not None and date_col in cand.columns and pd.api.types.is_datetime64_any_dtype(cand[date_col]):
                    same = cand[cand[date_col].dt.month == tgt.month]
                    fallback_note = f"(dựa trên cùng tháng)"
                else:
                    same = cand
                    fallback_note = ""

            lines = [f'Dự báo sơ bộ cho {time_desc} dựa trên dữ liệu lịch sử ({len(same)} mẫu{ " — " + fallback_note if fallback_note else "" }):']
            any_found = False
            for key, prefs in targets.items():
                col = _pick_col(prefs)
                if col and col in same.columns:
                    s = pd.to_numeric(same[col], errors='coerce').dropna()
                    if len(s) > 0:
                        any_found = True
                        m = s.mean()
                        sd = s.std()
                        if key == 'temp':
                            lines.append(f"- Nhiệt độ (trung bình): {m:.1f}°C (±{sd:.1f}), dựa trên {len(s)} mẫu")
                        elif key == 'pm25':
                            lines.append(f"- PM2.5 (trung bình): {m:.0f} µg/m3 (±{sd:.0f}), dựa trên {len(s)} mẫu")
                        elif key == 'pm10':
                            lines.append(f"- PM10 (trung bình): {m:.0f} µg/m3 (±{sd:.0f}), dựa trên {len(s)} mẫu")
                        elif key == 'aqi':
                            lines.append(f"- AQI (trung bình): {m:.0f} (±{sd:.0f}), dựa trên {len(s)} mẫu")

            if not any_found:
                # Try statistical spatial borrowing (correlation-based neighbors) as a fallback
                try:
                    # determine a StationId candidate in the data
                    station_candidate = None
                    if 'StationId' in cand.columns:
                        try:
                            station_candidate = cand['StationId'].mode().iloc[0]
                        except Exception:
                            station_candidate = None
                    # attempt neighbors if a station id is available
                    if station_candidate is not None:
                        neigh = find_statistical_neighbors(str(csv_path), station_candidate, metric_tokens=('pm25','pm10'), min_overlap=90, top_k=4)
                        if neigh:
                            wprob, details = weighted_neighbor_rain_prob(str(csv_path), station_candidate, neigh)
                            if wprob is not None:
                                lines.append(f"- Khả năng mưa (ước lượng từ các trạm lân cận thống kê): ~{int(wprob)}%")
                                lines.append('\nLưu ý: ước lượng này sử dụng các trạm có tương quan thống kê; độ tin cậy tuỳ thuộc vào mức tương quan và độ phủ mẫu.')
                                # show neighbor summary in debug session_state for inspect
                                try:
                                    st.session_state['neighbor_details'] = details
                                except Exception:
                                    pass
                                return '\n'.join(lines)
                except Exception:
                    pass

                # Try OpenWeather as a quick external fallback when no pollutant/temp columns found
                # Try server-side OpenWeather endpoint as fallback
                try:
                    ai_base = st.session_state.get('ai_url') or DEFAULT_AI_URL
                    ow_url = ai_base.replace('/ask', '/weather/ow-forecast')
                    params = {'city': city or os.getenv('DEFAULT_CITY', 'Hanoi')}
                    if 'tgt' in locals() and tgt is not None:
                        try:
                            params['date'] = pd.to_datetime(tgt).strftime('%Y-%m-%d')
                        except Exception:
                            pass
                    r = requests.get(ow_url, params=params, timeout=6)
                    if r.ok:
                        j = r.json()
                        # If detail contains pop (probability of precipitation), prefer to present it explicitly
                        pop = None
                        try:
                            detail = j.get('detail') or {}
                            pop = detail.get('daily_match', {}).get('pop')
                        except Exception:
                            pop = None

                        if pop is not None:
                            lines.append(f"- Khả năng mưa vào {time_desc}: ~{int(pop*100)}% (theo OpenWeather One Call)")
                            lines.append('\nLưu ý: xác suất mưa trên là dự báo mô hình (OpenWeather).')
                            return '\n'.join(lines)

                        if j.get('success') and j.get('summary'):
                            lines.append('\n' + j.get('summary'))
                            lines.append('\nLưu ý: dữ liệu OpenWeather (server-side) được dùng làm tham khảo tạm thời.')
                            return '\n'.join(lines)
                except Exception:
                    pass

                return f'Có {len(cand)} bản ghi lịch sử cho {time_desc}, nhưng không tìm thấy các cột phù hợp để ước lượng (ví dụ PM2.5/PM10/AQI).'

            # simple confidence heuristic
            total_n = sum([int(re.search(r"(\d+)$", l).group(1)) if re.search(r"(\d+)$", l) else 0 for l in lines[1:]])
            if total_n >= 30:
                conf = 'Cao'
            elif total_n >= 10:
                conf = 'Trung bình'
            else:
                conf = 'Thấp'
            lines.append(f'Độ tin cậy ước tính: {conf}. Lưu ý: dự báo này là climatology (không phản ánh mô hình thời tiết hiện tại).')
            return '\n'.join(lines)

        # Build summary text
        lines = [f'Dự báo sơ bộ cho {time_desc} dựa trên dữ liệu lịch sử ({len(cand)} mẫu{ " — cùng ngày qua các năm" if not fallback else " — cùng tháng" }):']
        for k, (mean, sd, n) in metrics.items():
            pretty = k
            if 'temp' in k:
                lines.append(f"- Nhiệt độ (trung bình): {mean:.1f}°C (±{sd:.1f}), dựa trên {n} mẫu")
            elif 'pm25' in k or 'pm2' in k:
                lines.append(f"- PM2.5 (trung bình): {mean:.0f} µg/m3 (±{sd:.0f}), dựa trên {n} mẫu")
            elif 'pm10' in k:
                lines.append(f"- PM10 (trung bình): {mean:.0f} µg/m3 (±{sd:.0f}), dựa trên {n} mẫu")
            elif 'aqi' in k:
                lines.append(f"- AQI (trung bình): {mean:.0f} (±{sd:.0f}), dựa trên {n} mẫu")
            else:
                lines.append(f"- {pretty}: {mean:.2f} (±{sd:.2f}) — {n} mẫu")

        # Confidence heuristic
        total_n = sum(v[2] for v in metrics.values())
        if total_n >= 30:
            conf = 'Cao'
        elif total_n >= 10:
            conf = 'Trung bình'
        else:
            conf = 'Thấp'

        # Rain probability and pollution risk estimation based on recent samples
        rain_prob = None
        rain_n = 0
        try:
            # Prefer numeric precipitation columns if present
            precip_cols = [c for c in cand.columns if any(x in c.lower() for x in ('rain', 'precip'))]
            if precip_cols:
                # use first precip column
                pcol = precip_cols[0]
                precip = pd.to_numeric(cand[pcol], errors='coerce').fillna(0)
                # count days with measurable precipitation (>0.5 mm)
                rain_n = int((precip > 0.5).sum())
                rain_prob = 100.0 * rain_n / len(precip) if len(precip) > 0 else None
            else:
                # fallback: look for description/text fields and search for 'mưa'/'rain'
                desc_cols = [c for c in cand.columns if any(x in c.lower() for x in ('desc', 'weather', 'note'))]
                if desc_cols:
                    dc = cand[desc_cols[0]].astype(str).fillna('').str.lower()
                    rain_n = int(dc.str.contains('mua|mưa|rain').sum())
                    rain_prob = 100.0 * rain_n / len(dc) if len(dc) > 0 else None
        except Exception:
            rain_prob = None

        # Pollution risk: fraction of days with PM2.5 above a threshold (e.g. 35 µg/m3)
        pollution_prob = None
        pollution_n = 0
        try:
            pm_cols = [c for c in cand.columns if any(x in c.lower() for x in ('pm2.5','pm25','pm2_5'))]
            if pm_cols:
                pm = pd.to_numeric(cand[pm_cols[0]], errors='coerce').dropna()
                pollution_n = int((pm > 35.4).sum())
                pollution_prob = 100.0 * pollution_n / len(pm) if len(pm) > 0 else None
        except Exception:
            pollution_prob = None

        lines.append(f'Độ tin cậy ước tính: {conf}. Lưu ý: phần trên là dựa trên trung bình lịch sử, bên dưới là dự đoán mô hình (SARIMA) nếu có thể tính được).')

        # Append rain/pollution summary
        if rain_prob is not None:
            lines.append(f"- Khả năng mưa vào {time_desc}: ~{rain_prob:.0f}% (dựa trên {len(cand)} mẫu; {rain_n} ngày có mưa trong cửa sổ).")
        else:
            lines.append(f"- Khả năng mưa vào {time_desc}: không đủ dữ liệu để ước lượng.")

        if pollution_prob is not None:
            lines.append(f"- Nguy cơ ô nhiễm PM2.5 (>35 µg/m3): ~{pollution_prob:.0f}% ({pollution_n}/{len(pm) if 'pm' in locals() else len(cand)} ngày).")
        else:
            lines.append("- Nguy cơ ô nhiễm: không đủ dữ liệu PM2.5 để ước lượng.")

        # Attempt model-based forecast (SARIMA) when we have enough historical daily series
        try:
            # build daily series grouped by date_col
            df_city_dates = df_city.copy()
            df_city_dates[date_col] = pd.to_datetime(df_city_dates[date_col])
            df_city_dates = df_city_dates.set_index(date_col)

            # determine steps to forecast
            if request_key == 'tomorrow':
                target_dt = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
            elif request_key == 'date' and params and params.get('date'):
                target_dt = pd.Timestamp(params.get('date')).normalize()
            elif request_key == 'range' and params and params.get('end'):
                target_dt = pd.Timestamp(params.get('end')).normalize()
            else:
                target_dt = None

                if target_dt is not None:
                    last_hist_date = df_city_dates.index.max().normalize()
                    steps = int((target_dt - last_hist_date).days)
                    if steps <= 0:
                        steps = 1

                    model_lines = ['\nDự đoán mô hình (SARIMA):']
                    # Try SARIMA for key metrics using last 30 days where possible
                    # build normalized map for df columns
                    df_norm_map = {re.sub(r'[^0-9a-z]', '', c.lower()): c for c in df_city_dates.columns}
                    for metric_col in ['temp', 'pm25', 'pm10', 'aqi']:
                        col = None
                        if metric_col in df_norm_map:
                            col = df_norm_map[metric_col]
                        else:
                            # fallback: look for a column whose normalized form contains the metric token
                            found = [c for n, c in df_norm_map.items() if metric_col in n]
                            if found:
                                col = found[0]
                        if not col:
                            continue
                        # build a recent 30-day series for the metric (prefer most recent days)
                        ser = pd.to_numeric(df_city_dates[col], errors='coerce')
                        # resample to daily mean to ensure regular spacing
                        try:
                            ser_daily = ser.resample('D').mean().dropna()
                        except Exception:
                            ser_daily = ser.dropna()

                        # focus on last 30 days
                        if len(ser_daily) > 30:
                            ser_recent = ser_daily.tail(30)
                        else:
                            ser_recent = ser_daily

                        if len(ser_recent) < 12:
                            model_lines.append(f"- {metric_col}: dữ liệu ít (n={len(ser_recent)}) — bỏ qua mô hình.")
                            continue
                        try:
                            mean, lower, upper = sarima_forecast(ser_recent, steps=steps)
                            pred_val = mean[-1]
                            lo = lower[-1]
                            hi = upper[-1]
                            if 'temp' in metric_col:
                                model_lines.append(f"- Nhiệt độ (SARIMA): {pred_val:.1f}°C (95% PI: {lo:.1f}–{hi:.1f}), mô hình huấn luyện trên {len(ser_recent)} mẫu")
                            elif 'pm25' in metric_col:
                                model_lines.append(f"- PM2.5 (SARIMA): {pred_val:.0f} µg/m3 (95% PI: {lo:.0f}–{hi:.0f}), mô hình huấn luyện trên {len(ser_recent)} mẫu")
                            elif 'pm10' in metric_col:
                                model_lines.append(f"- PM10 (SARIMA): {pred_val:.0f} µg/m3 (95% PI: {lo:.0f}–{hi:.0f}), mô hình huấn luyện trên {len(ser_recent)} mẫu")
                            elif 'aqi' in metric_col:
                                model_lines.append(f"- AQI (SARIMA): {pred_val:.0f} (95% PI: {lo:.0f}–{hi:.0f}), mô hình huấn luyện trên {len(ser_recent)} mẫu")
                        except Exception as me:
                            model_lines.append(f"- {metric_col}: mô hình thất bại ({me}).")

                if len(model_lines) > 1:
                    lines.extend(model_lines)
        except Exception:
            # don't fail on model errors
            pass

        lines.append('Hạn chế: dữ liệu lịch sử có thể không phản ánh các điều kiện thời tiết bất thường (tuyến bão, cold front, ô nhiễm đột biến).')
        return '\n'.join(lines)
    except Exception as e:
        return f'Lỗi khi phân tích dữ liệu lịch sử: {e}'

# Config
DEFAULT_AI_URL = os.getenv("AI_ASSISTANT_URL", "http://localhost:8000/ask")
TIMEOUT = 25

st.set_page_config(page_title="AI Assistant — Streamlit", page_icon="🤖", layout="wide")

def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "ai_url" not in st.session_state:
        st.session_state.ai_url = DEFAULT_AI_URL
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "cleaned_history" not in st.session_state:
        st.session_state.cleaned_history = []


init_state()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("AI Assistant — Giao diện kiểm thử")
    st.write("Gửi câu hỏi tới backend `ai-assistant` (thay cho Telegram/n8n).")
with col2:
    st.markdown("**Trạng thái**")
    try:
        health = requests.get(st.session_state.ai_url.replace('/ask', '/weather/health'), timeout=3).json()
        openweather_ok = health.get('apis', {}).get('openweather', False)
        waqi_ok = health.get('apis', {}).get('waqi', False)
        st.write("OpenWeather:", "✅" if openweather_ok else "❌")
        st.write("WAQI:", "✅" if waqi_ok else "❌")
    except Exception:
        st.write("AI service:", "❌ không kết nối")


with st.sidebar:
    st.header("Cấu hình")
    st.text_input("AI Assistant URL", value=st.session_state.ai_url, key="ai_url_input")
    if st.button("Áp dụng URL mới"):
        st.session_state.ai_url = st.session_state.ai_url_input.strip() or DEFAULT_AI_URL
        st.success("URL mới đã được áp dụng. Nếu cần, hãy tải lại trình duyệt để áp dụng thay đổi.")
    st.markdown("---")
    st.write("GDRIVE_FOLDER_ID: `./.env` trên server`")
    st.caption("Server sẽ quyết định lưu hay không dựa trên biến môi trường.")
    st.markdown("---")
    st.checkbox("Hiển thị debug (endpoint & payload)", key='debug_mode', value=False)
    st.markdown("---")
    st.write("NLP: nhận diện nhanh các câu hỏi về thời tiết. Nếu là truy vấn thời tiết, app sẽ gọi endpoint `/weather/save-clean` để lấy dữ liệu đã làm sạch và hiển thị biểu đồ.")


with st.form("ask_form"):
    user_input = st.text_area("Nhập câu hỏi hoặc yêu cầu", height=160, placeholder="Ví dụ: thời tiết HCM")
    show_raw = st.checkbox("Hiển thị JSON thô trả về", value=False)
    submitted = st.form_submit_button("Gửi")

    if submitted:
        if not user_input.strip():
            st.warning("Vui lòng nhập nội dung trước khi gửi.")
        else:
            # Classify query type and handle special analysis locally when possible
            qtype = detect_query_type(user_input)
            payload = {"message": {"text": user_input}}
            headers = {"Content-Type": "application/json"}

            # Handle local-only forecast queries (no backend call needed)
            if qtype == 'forecast':
                # detect explicit future date if provided
                try:
                    is_future, future_key, future_params = detect_future_query(user_input)
                except Exception:
                    is_future, future_key, future_params = (False, 'tomorrow', None)
                fk = future_key if is_future else 'tomorrow'
                city = extract_city(user_input)
                with st.spinner('Đang tạo dự báo từ dữ liệu lịch sử...'):
                    try:
                        forecast_text = forecast_from_csv(fk, city)
                        # show detected CSV columns for debugging (if available) inside an expander
                        try:
                            cols = st.session_state.get('detected_csv_columns')
                            if cols:
                                with st.expander('Cột CSV phát hiện (click để xem)'):
                                    # Build a small DataFrame for nicer display
                                    try:
                                        import pandas as _pd
                                        dfcols = _pd.DataFrame([{'normalized': k, 'column': v} for k, v in list(cols.items())])
                                        # show first 100 rows, provide copy box below
                                        st.table(dfcols.head(200))
                                        mapping_text = '\n'.join([f"{k} -> {v}" for k, v in cols.items()])
                                        st.text_area('Copy mapping (Ctrl+C):', value=mapping_text, height=120)
                                    except Exception:
                                        st.write(', '.join([f"{k} -> {v}" for k, v in cols.items()]))
                        except Exception:
                            pass
                        record = {
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "question": user_input,
                            "reply": forecast_text,
                            "raw": {},
                            "called_url": 'local:forecast',
                            "elapsed": 0.0
                        }
                        st.session_state.history.insert(0, record)
                        st.session_state.last_response = record
                        st.success(forecast_text)
                    except Exception as e:
                        st.error(f"Lỗi khi tạo dự báo: {e}")
                # skip the backend call; stop this run so the UI shows the local result
                st.stop()

            # Handle local trend/compare queries when possible
            if qtype in ('trend', 'compare'):
                try:
                    from pathlib import Path
                    csv_path = Path.cwd() / 'data' / 'cleaned' / 'station_day_clean.csv'
                    if not csv_path.exists():
                        st.error('Không có file lịch sử (station_day_clean.csv) để phân tích.')
                        st.stop()
                    df_hist = pd.read_csv(csv_path, low_memory=False)
                    df_hist.columns = [c.strip() for c in df_hist.columns]
                    if qtype == 'trend':
                        city = extract_city(user_input)
                        dfc = df_hist[df_hist.get('city', '').astype(str).str.lower() == city.lower()] if 'city' in df_hist.columns else df_hist
                        if dfc.empty:
                            st.info('Không có dữ liệu lịch sử cho thành phố này để phân tích xu hướng.')
                            st.stop()
                        # try to use pm25/temp columns
                        dfc['Date'] = pd.to_datetime(dfc[dfc.columns[0]], errors='coerce') if 'timestamp' not in dfc.columns else pd.to_datetime(dfc['timestamp'], errors='coerce')
                        dfc = dfc.sort_values('Date').dropna(subset=['Date'])
                        # use last 30 days
                        tail = dfc.tail(30)
                        texts = []
                        figs = []
                        for metric in ('pm25', 'temp'):
                            if metric in tail.columns:
                                ser = pd.to_numeric(tail[metric], errors='coerce').dropna()
                                if len(ser) >= 3:
                                    # simple slope estimate
                                    x = (pd.to_datetime(tail['Date']).map(pd.Timestamp.toordinal).values[-len(ser):])
                                    y = ser.values[-len(ser):]
                                    try:
                                        coef = ( ( (x - x.mean()) * (y - y.mean()) ).sum() ) / ( ((x - x.mean())**2).sum() )
                                        trend = 'tăng' if coef > 0 else 'giảm' if coef < 0 else 'ổn định'
                                        texts.append(f'Xu hướng gần đây cho {metric}: {trend} (ước lượng slope={coef:.4f})')
                                    except Exception:
                                        texts.append(f'Không thể ước lượng xu hướng cho {metric}.')
                                    # small plot
                                    fig = px.line(tail, x='Date', y=metric, title=f'Xu hướng {metric} — {city}')
                                    figs.append(fig)
                        reply_text = '\n'.join(texts) if texts else 'Không có chỉ số phù hợp để phân tích xu hướng.'
                        record = {
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'question': user_input,
                            'reply': reply_text,
                            'raw': {},
                            'called_url': 'local:trend',
                            'elapsed': 0.0
                        }
                        st.session_state.history.insert(0, record)
                        st.session_state.last_response = record
                        st.success(reply_text)
                        for f in figs:
                            st.plotly_chart(f, use_container_width=True)
                        st.stop()
                    if qtype == 'compare':
                        cities = extract_cities_for_compare(user_input)
                        if len(cities) < 2:
                            st.info('Vui lòng nêu rõ hai thành phố để so sánh (ví dụ: "so sánh Hanoi và HCM").')
                            st.stop()
                        # compute simple means for each city
                        metrics = ['temp','pm25','pm10','aqi']
                        rows = []
                        for c in cities[:2]:
                            dfc = df_hist[df_hist.get('city', '').astype(str).str.lower() == c.lower()] if 'city' in df_hist.columns else df_hist
                            if dfc.empty:
                                rows.append({'city': c, 'note': 'no data'})
                                continue
                            row = {'city': c}
                            for m in metrics:
                                if m in dfc.columns:
                                    row[m] = float(pd.to_numeric(dfc[m], errors='coerce').dropna().mean() or 0)
                                else:
                                    row[m] = None
                            rows.append(row)
                        comp_df = pd.DataFrame(rows)
                        st.write('So sánh trung bình (gần nhất):')
                        st.table(comp_df)
                        # bar chart for PM2.5 if available
                        if 'pm25' in comp_df.columns:
                            figc = px.bar(comp_df, x='city', y='pm25', title='So sánh PM2.5 trung bình')
                            st.plotly_chart(figc, use_container_width=True)
                        record = {
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'question': user_input,
                            'reply': 'So sánh đã được hiển thị.',
                            'raw': {},
                            'called_url': 'local:compare',
                            'elapsed': 0.0
                        }
                        st.session_state.history.insert(0, record)
                        st.session_state.last_response = record
                        st.stop()
                except Exception as e:
                    st.error(f'Lỗi khi phân tích dữ liệu lịch sử: {e}')
                    st.stop()

            # Otherwise fall back to server (ask endpoint) as before
            with st.spinner("Đang gửi yêu cầu..."):
                start = time.time()
                try:
                    is_weather = detect_weather_intent(user_input)
                    called_url = None
                    if is_weather:
                        city = extract_city(user_input)
                        save_clean_url = st.session_state.ai_url.replace('/ask', '/weather/save-clean')
                        called_url = f"{save_clean_url}?city={requests.utils.quote(city)}"
                        # call backend to fetch cleaned data and optionally save to Drive
                        resp = requests.post(called_url, timeout=TIMEOUT)
                    else:
                        called_url = st.session_state.ai_url
                        resp = requests.post(st.session_state.ai_url, json=payload, headers=headers, timeout=TIMEOUT)
                    elapsed = time.time() - start
                    resp.raise_for_status()
                    data = resp.json()
                    # Normalize reply field
                    reply = data.get("reply") or data.get("result") or data.get("ad_text") or data.get("message")
                    if not reply:
                        # Try common fallbacks
                        reply = data.get("response") or data.get("text") or None

                    # If weather intent, the backend returns cleaned JSON (not a reply string).
                    # Build a human-readable reply from the cleaned payload.
                    if is_weather and isinstance(data, dict) and data.get('cleaned'):
                        cleaned = data['cleaned']
                        w = cleaned.get('weather', {}) or {}
                        a = cleaned.get('aqi', {}) or {}
                        analysis = cleaned.get('analysis', {}) or {}
                        temp = w.get('temp')
                        feels = w.get('feels_like')
                        hum = w.get('humidity')
                        desc = w.get('description')
                        aqi_val = analysis.get('aqi') or a.get('aqi')
                        aqi_cat = analysis.get('aqi_category') or a.get('source')
                        recommendation = analysis.get('recommendation') or ''
                        # fix possible mojibake in recommendation
                        recommendation = fix_text_encoding(recommendation)

                        # Build reply_text using randomized templates for variety
                        city_name = cleaned.get('city', city)
                        pm25_val = a.get('pm25') if isinstance(a, dict) else None
                        pm10_val = a.get('pm10') if isinstance(a, dict) else None
                        aqi_display = aqi_val if aqi_val is not None else (a.get('aqi') if isinstance(a, dict) else None)

                        # Fallback text pieces
                        temp_s = f"{temp}°C" if temp is not None else 'không có dữ liệu nhiệt độ'
                        feels_s = f"(cảm giác {feels}°C)" if feels is not None else ''
                        hum_s = f"Độ ẩm {hum}%" if hum is not None else ''
                        desc_s = desc or ''
                        aqi_s = f"AQI {aqi_display}" if aqi_display is not None else ''
                        pm25_s = f"PM2.5: {pm25_val}" if pm25_val is not None else ''
                        pm10_s = f"PM10: {pm10_val}" if pm10_val is not None else ''

                        templates = [
                            "Ở {city}, {desc}. Nhiệt độ khoảng {temp} {feels}. {hum} {aqi_info} {advice}",
                            "Thông tin {city}: {desc}. {temp}, {hum}. {aqi_info}; {pm25} {pm10} {advice}",
                            "Tóm tắt tại {city}: {desc} — {temp} {feels}. {aqi_info} {advice}",
                            "Bản tin nhanh ({city}): {desc}. {temp} — {hum}. {aqi_info}. {advice}"
                        ]

                        tmpl = random.choice(templates)
                        aqi_info = (f"AQI: {aqi_display} ({aqi_cat})" if aqi_display is not None else '')
                        advice = recommendation or ''
                        reply_text = tmpl.format(city=city_name, desc=desc_s, temp=temp_s, feels=feels_s, hum=hum_s, aqi_info=aqi_info, pm25=pm25_s, pm10=pm10_s, advice=advice).strip()
                        # clean repeated spaces and stray punctuation
                        reply_text = re.sub(r'\s{2,}', ' ', reply_text)
                        reply_text = re.sub(r'\s+\.', '.', reply_text)

                        # fix mojibake in reply_text
                        reply_text = fix_text_encoding(reply_text)

                        # If user asked about a future day, try to produce a CSV-based forecast
                        try:
                            is_future, future_key = detect_future_query(user_input)
                            if is_future:
                                forecast_text = forecast_from_csv(future_key, cleaned.get('city') or city)
                                reply_text += "\n\n" + forecast_text
                        except Exception:
                            # don't break main flow if forecast helper fails
                            pass

                        # Extract pollutant concentrations from cleaned data (robust to different schemas)
                        pollutants = {}
                        # Try cleaned['aqi'] first
                        try:
                            aq = data.get('cleaned', {}).get('aqi', {}) or {}
                            for k in ['pm25', 'pm_25', 'pm2_5', 'pm2.5', 'pm10', 'no2', 'so2', 'o3', 'co']:
                                if k in aq and aq.get(k) is not None:
                                    pollutants[k] = aq.get(k)
                        except Exception:
                            pass

                        # Try cleaned.raw.components (OpenWeather style)
                        try:
                            raw = data.get('cleaned', {}).get('raw', {}) or {}
                            comps = raw.get('components') or {}
                            for k, v in comps.items():
                                # normalize names like 'pm2_5' -> 'pm25'
                                kn = k.replace('.', '').replace('_', '').lower()
                                if kn in ('pm25', 'pm2_5', 'pm2.5'):
                                    pollutants['pm25'] = v
                                elif kn in ('pm10',):
                                    pollutants['pm10'] = v
                                elif kn in ('no2',):
                                    pollutants['no2'] = v
                                elif kn in ('so2',):
                                    pollutants['so2'] = v
                                elif kn in ('o3',):
                                    pollutants['o3'] = v
                                elif kn in ('co',):
                                    pollutants['co'] = v
                        except Exception:
                            pass

                        # Additional fallbacks: top-level cleaned fields
                        try:
                            cleaned_top = data.get('cleaned', {})
                            for label in ['PM2.5', 'PM10', 'pm25', 'pm10', 'AQI']:
                                if label in cleaned_top and cleaned_top.get(label) is not None:
                                    pollutants[label.lower()] = cleaned_top.get(label)
                        except Exception:
                            pass

                        record = {
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "question": user_input,
                            "reply": reply_text,
                            "raw": data,
                            "called_url": called_url,
                            "pollutants": pollutants,
                            "elapsed": round(elapsed, 2)
                        }
                        st.session_state.history.insert(0, record)
                        st.session_state.last_response = record

                        # also store cleaned record for charts
                        flat = {
                            'timestamp': cleaned.get('timestamp_utc') or cleaned.get('timestamp') or time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                            'city': cleaned.get('city') or city,
                            'temp': w.get('temp'),
                            'pm25': a.get('pm25'),
                            'aqi': analysis.get('aqi') or a.get('aqi'),
                            'aqi_category': analysis.get('aqi_category')
                        }
                        st.session_state.cleaned_history.insert(0, flat)
                    else:
                        # Non-weather replies: use server-provided text
                        if reply and isinstance(reply, str):
                            reply = fix_text_encoding(reply)

                        record = {
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "question": user_input,
                            "reply": reply,
                            "raw": data,
                            "called_url": called_url,
                            "elapsed": round(elapsed, 2)
                        }
                        st.session_state.history.insert(0, record)
                        st.session_state.last_response = record
                except requests.exceptions.RequestException as e:
                    st.error(f"Lỗi khi gọi API: {e}")
                    st.session_state.history.insert(0, {"timestamp": time.strftime('%Y-%m-%d %H:%M:%S'), "question": user_input, "reply": None, "raw": {"error": str(e)}})


st.markdown("---")
left, right = st.columns([2, 1])

with left:
    st.subheader("Kết quả mới nhất")
    if st.session_state.last_response:
        lr = st.session_state.last_response
        if lr.get('reply'):
            st.markdown("**Phản hồi:**")
            st.success(lr['reply'])
            # show debug info if enabled (avoid deep nested indentation that sometimes
            # caused parsing issues inside some runtimes)
            debug_mode = bool(st.session_state.get('debug_mode'))
            if debug_mode:
                called = lr.get('called_url') or 'n/a'
                st.caption(f"Endpoint gọi: {called}")
                st.subheader('JSON thô (debug)')
                st.json(lr.get('raw'))
            # If pollutant concentrations were captured, display them and a small interpretation
            pollutants = lr.get('pollutants') or {}
            if pollutants:
                st.markdown('**Nồng độ các chất (hiện tại)**')
                # Normalize to nicer keys for display
                nice = {k: v for k, v in pollutants.items()}
                try:
                    st.table(nice)
                except Exception:
                    st.write(nice)

                # Build a small interpretation block
                interp_lines = []
                for k, v in nice.items():
                    cat, advice = categorize_pollutant(k, v)
                    if cat:
                        interp_lines.append(f"- {k.upper()}: {v} → {cat}. {advice}")
                if interp_lines:
                    with st.expander('Phân tích chất lượng không khí'):
                        for line in interp_lines:
                            st.write(line)
        # Auto-show map: if the latest cleaned response contains coordinates, render map here
        map_shown = False
        cleaned_for_map = None
        coord_for_map = None
        try:
            raw_for_map = lr.get('raw') or {}
            cleaned_for_map = raw_for_map.get('cleaned') if isinstance(raw_for_map, dict) else None
            if cleaned_for_map:
                coord_for_map = extract_coords_from_cleaned(cleaned_for_map)
            if coord_for_map:
                lat_m = coord_for_map.get('lat')
                lon_m = coord_for_map.get('lon')
                try:
                    latf = float(lat_m)
                    lonf = float(lon_m)
                    st.markdown('---')
                    st.subheader('Bản đồ (vị trí phản hồi mới nhất)')
                    show_map(latf, lonf, cleaned_for_map)
                    map_shown = True
                except Exception:
                    map_shown = False
        except Exception:
            map_shown = False

        # Only show the generic warning when there is no textual reply AND no map was shown
        if not lr.get('reply') and not map_shown:
            st.warning("Không nhận được phản hồi văn bản từ server. Xem JSON thô bên dưới.")

        st.caption(f"Thời gian: {lr.get('elapsed', '?')}s — {lr.get('timestamp')}")
        if show_raw:
            st.subheader("JSON thô")
            st.json(lr.get('raw'))
    # Visualization area: show charts if we have cleaned history
    if st.session_state.cleaned_history:
        st.markdown("---")
        st.subheader("Biểu đồ thời tiết (dữ liệu đã làm sạch)")
        try:
            df = pd.DataFrame(st.session_state.cleaned_history)
            # normalize and parse
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # numeric columns
            for c in ['temp', 'pm25', 'pm10', 'aqi']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

            # Sidebar controls for charts (use explicit keys so values persist across reruns)
            metrics = st.sidebar.multiselect("Chọn biểu đồ hiển thị", options=['temp', 'humidity', 'pm25', 'pm10', 'aqi'], default=['temp','aqi','pm25'], key='chart_metrics')
            # Avoid Streamlit slider error when there's only one data point
            max_points = max(1, len(df))
            default_n = min(10, len(df))
            if max_points == 1:
                last_n = 1
            else:
                last_n = st.sidebar.slider('Số điểm hiển thị (mới nhất)', min_value=1, max_value=max_points, value=default_n, key='chart_n')
            animated = st.sidebar.checkbox('Biểu đồ động (animated)', value=False, key='chart_animated')
            plot_df = df.sort_values('timestamp').tail(last_n)

            if animated and len(plot_df) > 1:
                # Animated charts disabled: use static charts to avoid runtime indentation/animation issues
                st.info("Biểu đồ động hiện tạm thời bị vô hiệu hóa. Hãy bỏ chọn 'Biểu đồ động' để xem biểu đồ tĩnh.")
            else:
                # Build nicer Plotly stacked charts for readability
                selected = [m for m in metrics if m in plot_df.columns]
                if not selected:
                    st.info('Không có dữ liệu phù hợp cho biểu đồ đã chọn.')
                else:
                    rows = len(selected)
                    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                        subplot_titles=[{'temp':'Nhiệt độ','humidity':'Độ ẩm','pm25':'PM2.5','pm10':'PM10','aqi':'AQI'}.get(m,m) for m in selected])

                    for i, metric in enumerate(selected, start=1):
                        if metric == 'temp' and metric in plot_df:
                            fig.add_trace(go.Scatter(
                                x=plot_df['timestamp'], y=plot_df['temp'], mode='lines+markers', name='Temp',
                                line=dict(color='#ff7f0e'), fill='tozeroy', hovertemplate='%{x}<br>Temp: %{y}°C'
                            ), row=i, col=1)

                        elif metric == 'humidity' and 'humidity' in plot_df:
                            fig.add_trace(go.Scatter(
                                x=plot_df['timestamp'], y=plot_df['humidity'], mode='lines+markers', name='Humidity',
                                line=dict(color='#17becf'), hovertemplate='%{x}<br>Humidity: %{y}%'
                            ), row=i, col=1)

                        elif metric == 'pm25' and 'pm25' in plot_df:
                            # color bars by threshold
                            colors = [ '#00c853' if (v is not None and v<=12) else '#ffeb3b' if (v is not None and v<=35.4) else '#ff9100' if (v is not None and v<=55.4) else '#f44336' if (v is not None and v<=150.4) else '#9c27b0' if (v is not None and v<=250.4) else '#6d0000' for v in plot_df['pm25'] ]
                            fig.add_trace(go.Bar(
                                x=plot_df['timestamp'], y=plot_df['pm25'], name='PM2.5', marker_color=colors,
                                hovertemplate='%{x}<br>PM2.5: %{y} µg/m3'
                            ), row=i, col=1)

                        elif metric == 'pm10' and 'pm10' in plot_df:
                            fig.add_trace(go.Bar(
                                x=plot_df['timestamp'], y=plot_df['pm10'], name='PM10', marker_color='#1b9e77',
                                hovertemplate='%{x}<br>PM10: %{y} µg/m3'
                            ), row=i, col=1)

                        elif metric == 'aqi' and 'aqi' in plot_df:
                            aqi_colors = [ _aqi_color(v) for v in plot_df['aqi'] ]
                            fig.add_trace(go.Scatter(
                                x=plot_df['timestamp'], y=plot_df['aqi'], mode='lines+markers', name='AQI',
                                marker=dict(color=aqi_colors, size=8), line=dict(color='#7f3b8b'),
                                hovertemplate='%{x}<br>AQI: %{y}'
                            ), row=i, col=1)

                    # Layout polish
                    fig.update_layout(height=220*rows, showlegend=False, template='plotly_white', margin=dict(t=40,b=40,l=40,r=40))
                    fig.update_xaxes(title_text='Thời gian')
                    # persist the last constructed figure so it survives reruns
                    st.session_state['last_fig'] = fig
                    st.plotly_chart(st.session_state['last_fig'], width='stretch')
        except Exception as e:
            st.error(f"Không thể vẽ biểu đồ: {e}")
    else:
        st.info("Chưa có phản hồi nào. Gửi câu hỏi để bắt đầu.")

    st.markdown("---")
    st.subheader("Lịch sử tương tác")
    for i, item in enumerate(st.session_state.history):
        with st.expander(f"#{i+1} — {item['timestamp']}"):
            st.write("**Câu hỏi:**", item['question'])
            if item.get('reply'):
                st.write("**Phản hồi:**")
                st.success(item['reply'])
            else:
                st.write("**Phản hồi:** (không có)")
            if st.checkbox(f"Xem JSON thô cho #{i+1}", key=f"raw_{i}"):
                st.json(item.get('raw'))

    # --- Bản đồ (Folium) ---
    st.markdown("---")
    st.subheader("Bản đồ vị trí")
    st.write("Hiển thị vị trí và dữ liệu khí tượng/ô nhiễm cho bản ghi mới nhất hoặc theo tên thành phố.")
    col_map_left, col_map_right = st.columns([3,1])
    with col_map_right:
        city_input = st.text_input("Tìm theo thành phố (tùy chọn)", value="")
        if st.button("Lấy dữ liệu và hiển thị bản đồ"):
            # call backend to fetch cleaned data for city
            target_city = city_input.strip() or None
            if not target_city and st.session_state.last_response:
                # try to deduce city from last response
                raw = st.session_state.last_response.get('raw') or {}
                cleaned = raw.get('cleaned') if isinstance(raw, dict) else None
                target_city = (cleaned.get('city') if cleaned else None) or ''
            if not target_city:
                st.error('Vui lòng nhập tên thành phố hoặc có một phản hồi trước đó để lấy tọa độ.')
            else:
                save_clean_url = st.session_state.ai_url.replace('/ask', '/weather/save-clean')
                called = f"{save_clean_url}?city={requests.utils.quote(target_city)}"
                try:
                    r = requests.post(called, timeout=TIMEOUT)
                    r.raise_for_status()
                    d = r.json()
                    cleaned = d.get('cleaned') or {}
                    # try to find coords
                    coord = extract_coords_from_cleaned(cleaned)
                    if not coord:
                        st.error('Không tìm thấy tọa độ trong phản hồi. Mở JSON thô để kiểm tra trường `coord` hoặc `raw.coord`.')
                    else:
                        lat = coord.get('lat')
                        lon = coord.get('lon')
                        try:
                            latf = float(lat)
                            lonf = float(lon)
                            with col_map_left:
                                show_map(latf, lonf, cleaned)
                        except Exception:
                            st.error('Tọa độ không hợp lệ (không thể chuyển sang float).')
                except Exception as e:
                    st.error(f'Không thể lấy dữ liệu từ server: {e}')

    # Quick map from last response
    if st.button('Hiển thị bản đồ cho phản hồi mới nhất'):
        lr = st.session_state.last_response
        if not lr:
            st.error('Chưa có phản hồi nào.')
        else:
            raw = lr.get('raw') or {}
            cleaned = raw.get('cleaned') if isinstance(raw, dict) else None
            if not cleaned:
                st.error('Phản hồi mới nhất không chứa dữ liệu đã làm sạch.')
            else:
                coord = extract_coords_from_cleaned(cleaned)
                if not coord:
                    st.error('Không tìm thấy tọa độ trong dữ liệu đã làm sạch.')
                else:
                    try:
                        latf = float(coord.get('lat'))
                        lonf = float(coord.get('lon'))
                        show_map(latf, lonf, cleaned)
                    except Exception:
                        st.error('Tọa độ không hợp lệ trong bản ghi.')

with right:
    st.subheader("Trạng thái server")
    try:
        health = requests.get(st.session_state.ai_url.replace('/ask', '/weather/health'), timeout=2).json()
        st.write("OpenWeather:", "✅" if health.get('apis', {}).get('openweather') else "❌")
        st.write("WAQI:", "✅" if health.get('apis', {}).get('waqi') else "❌")
    except Exception:
        st.error("Không thể kết nối tới server. Kiểm tra `AI Assistant URL` và container.`")

    st.markdown("---")
    st.caption("Ghi chú: giao diện này chỉ gửi yêu cầu tới server. Việc lưu lên Google Drive do server quyết định dựa trên biến môi trường.")

st.markdown("---")
st.caption("Ghi chú: để lưu lên Google Drive, cấu hình `GDRIVE_FOLDER_ID` và `GOOGLE_APPLICATION_CREDENTIALS` trên server `ai-assistant`. Streamlit ở client sẽ chỉ gửi yêu cầu tới server và server sẽ thực hiện lưu.")
