#!/usr/bin/env python3
"""Run the 20 user-provided tests against the local ai-assistant service.
Saves full responses to `scripts/test_results.json` and prints a summary table.
"""
import requests
import time
import json
import unicodedata
import os

AI_URL = os.getenv('AI_ASSISTANT_URL', 'http://localhost:8000')
TIMEOUT = 25

tests = [
    "Thời tiết ở Hồ Chí Minh ngày mai như thế nào? PM2.5 khoảng bao nhiêu?",
    "Dự báo PM2.5 và AQI cho Hà Nội vào ngày 25/11?",
    "Ngày 01-12-2025 ở Đà Nẵng có mưa không? PM10 sẽ ra sao?",
    "Tuần tới ở Huế, ô nhiễm không khí có tăng không (PM2.5, O3)?",
    "Sáng mai ở Cần Thơ nhiệt độ và độ ẩm dự kiến bao nhiêu?",
    "So sánh AQI giữa Hồ Chí Minh và Hà Nội vào ngày mai.",
    "Dựa trên dữ liệu lịch sử, dự đoán PM2.5 cho Nha Trang ngày 25/11.",
    "Hôm nay ở Hải Phòng nồng độ NO2 và SO2 là bao nhiêu, và có nguy hiểm không?",
    "Cho tôi biết PM2.5 trung bình trong tháng trước ở Buôn Ma Thuột.",
    "Ngày mai có cần đeo khẩu trang ở Hồ Chí Minh không (dựa trên PM2.5/AQI)?",
    "Dự báo ô nhiễm (PM2.5, PM10, O3) cho Hà Nội trong 3 ngày tới.",
    "Ngày 26/11/2025 buổi chiều ở TP. HCM CO sẽ ở mức nào?",
    "Tuần này ở Hà Nội có đợt ô nhiễm cao không? Nếu có, khi nào khả năng cao nhất?",
    "Dự báo nhiệt độ cao nhất và thấp nhất cho Đà Nẵng ngày mai.",
    "Cho biết xu hướng AQI ở Hồ Chí Minh trong vòng 7 ngày gần nhất (tăng/giảm?).",
    "Nếu PM2.5 > 100 tại Hà Nội, lời khuyên ngắn gọn cho người già và trẻ em là gì?",
    "Tôi muốn lịch sử PM10 cho Cần Thơ 30 ngày gần nhất (trả về trung bình, max, min).",
    "Ngày 25/11, O3 ở Huế có thể vượt ngưỡng nguy hại không? (hãy cho PI nếu có thể)",
    "Dự báo tổng quan cho Sài Gòn (Ho Chi Minh) ngày 2025-12-01: nhiệt độ, mưa, AQI, khuyến nghị.",
    "So sánh ô nhiễm PM2.5 giữa ba thành phố: Hà Nội, Hồ Chí Minh và Đà Nẵng cho ngày mai."
]


def _norm(s: str) -> str:
    if not s:
        return ''
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch)).lower()


def detect_weather_intent(text: str) -> bool:
    text_norm = _norm(text)
    weather_keywords = ["thoi tiet", "thời tiết", "nhiet do", "nhiệt độ", "mưa", "nắng", "gió", "gio", "aqi", "o nhiem", "ô nhiễm", "pm2", "pm2.5", "pm10"]
    city_tokens = ["hcm", "ho chi minh", "hochiminh", "saigon", "sai gon", "hanoi", "ha noi", "da nang", "danang", "hue", "can tho", "cantho"]
    has_weather_kw = any(k in text_norm for k in weather_keywords)
    has_city = any(c in text_norm for c in city_tokens)
    date_indicators = ['mai', 'ngay', 'ngày', 'tuần', 'tuan', 'hôm', 'hom']
    has_date = any(d in text_norm for d in date_indicators)
    if has_weather_kw:
        return True
    if has_city and has_date:
        return True
    return False


def extract_city(text: str) -> str:
    t = _norm(text)
    if any(tok in t for tok in ["hcm", "ho chi minh", "hochiminh", "saigon", "sai gon"]):
        return 'Ho Chi Minh'
    if any(tok in t for tok in ["hanoi", "ha noi"]):
        return 'Hanoi'
    if any(tok in t for tok in ["da nang", "danang"]):
        return 'Da Nang'
    if 'hue' in t:
        return 'Hue'
    if 'can tho' in t or 'cantho' in t:
        return 'Can Tho'
    if 'hai phong' in t or 'haiphong' in t:
        return 'Hai Phong'
    if 'nha trang' in t or 'nhatrang' in t:
        return 'Nha Trang'
    if 'buon ma thuot' in t or 'buonmathuot' in t:
        return 'Buon Ma Thuot'
    return os.getenv('DEFAULT_CITY', 'Hanoi')


def run():
    results = []
    for i, q in enumerate(tests, start=1):
        is_weather = detect_weather_intent(q)
        called_url = None
        status = None
        summary = None
        resp_body = None
        elapsed = None
        start = time.time()
        try:
            if is_weather:
                city = extract_city(q)
                called_url = f"{AI_URL}/weather/save-clean?city={requests.utils.quote(city)}"
                r = requests.post(called_url, timeout=TIMEOUT)
            else:
                called_url = f"{AI_URL}/ask"
                payload = {"message": {"text": q}}
                r = requests.post(called_url, json=payload, timeout=TIMEOUT)
            elapsed = time.time() - start
            status = r.status_code
            try:
                resp_body = r.json()
                # try to extract a short reply
                if isinstance(resp_body, dict):
                    if resp_body.get('cleaned'):
                        cleaned = resp_body.get('cleaned')
                        w = cleaned.get('weather', {}) or {}
                        a = cleaned.get('aqi', {}) or {}
                        summary = f"cleaned: temp={w.get('temp')} aqi={a.get('aqi') or a.get('pm25')}"
                    else:
                        summary = resp_body.get('reply') or resp_body.get('response') or str(resp_body)[:120]
                else:
                    summary = str(resp_body)[:120]
            except Exception:
                resp_body = r.text
                summary = resp_body[:120]
        except Exception as e:
            elapsed = time.time() - start
            status = 'ERR'
            resp_body = {'error': str(e)}
            summary = str(e)

        results.append({
            'id': i,
            'question': q,
            'is_weather': is_weather,
            'called_url': called_url,
            'status': status,
            'elapsed_s': round(elapsed, 2) if elapsed else None,
            'summary': summary,
            'response': resp_body
        })
        print(f"[{i:02d}] status={status} time={round(elapsed or 0,2)}s weather={is_weather} -> {summary}")

    # save results
    out_path = os.path.join(os.path.dirname(__file__), 'test_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\nSaved full results to', out_path)


if __name__ == '__main__':
    run()
