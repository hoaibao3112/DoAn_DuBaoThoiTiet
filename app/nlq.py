import os
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.weather import (
    get_weather_data,
    get_aqi_data,
    get_aqi_category,
    fetch_raw_weather_and_aqi,
    clean_and_summarize_raw,
    generate_recommendation,
)

router = APIRouter()


# Simple module-level cache for the station-day dataframe
_DF_CACHE = None


def _load_station_df() -> pd.DataFrame:
    global _DF_CACHE
    if _DF_CACHE is not None:
        return _DF_CACHE

    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, "data", "cleaned", "station_day_clean.csv")
    if not os.path.exists(path):
        # try workspace root
        path = os.path.join(os.getcwd(), "data", "cleaned", "station_day_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"station_day_clean.csv not found at {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    _DF_CACHE = df
    return df


def _extract_top_n(df: pd.DataFrame, metric: str, top_n: int = 5, station_id: Optional[str] = None):
    if station_id:
        df = df[df["StationId"].astype(str).str.upper() == station_id.upper()]
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in data")
    out = df.sort_values(metric, ascending=False)[["Date", "StationId", metric]].head(top_n)
    return out.to_dict(orient="records")


def _monthly_average(df: pd.DataFrame, metric: str, station_id: Optional[str] = None):
    if station_id:
        df = df[df["StationId"].astype(str).str.upper() == station_id.upper()]
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in data")
    grp = df.groupby(df["Date"].dt.to_period("M"))[metric].mean().reset_index()
    grp["Date"] = grp["Date"].dt.to_timestamp()
    return grp.sort_values("Date").to_dict(orient="records")


@router.post("/nlq")
def nlq(payload: dict):
    """Basic Natural-Language Query handler (rule-based).

    Payload schema (JSON):
      - question: str (required)
      - station_id: optional station id like 'AP001'
      - metric: optional metric name like 'PM2.5', 'PM10', 'AQI'
      - top_n: optional int for top-k queries
    """
    try:
        question = (payload.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        q = question.lower()

        # 1) Current weather intent (keywords)
        if any(k in q for k in ["thời tiết", "nhiệt độ", "hôm nay", "nay", "weather"]):
            # optional city param
            city = payload.get("city") or payload.get("q_city") or "Ho Chi Minh"
            # Try to use the unified fetch->clean pipeline when available
            try:
                raw = fetch_raw_weather_and_aqi(city)
                cleaned = clean_and_summarize_raw(city, raw)
            except Exception:
                # Fallback to individual helpers
                weather = get_weather_data(city)
                aqi = get_aqi_data(city)
                aqi_val = int(aqi.get("aqi", 0)) if isinstance(aqi, dict) else 0
                cleaned = {
                    "weather": weather,
                    "aqi": aqi,
                    "analysis": {
                        "aqi": aqi_val,
                        "aqi_category": get_aqi_category(aqi_val),
                        "recommendation": generate_recommendation(aqi_val, weather.get("temp", 0), weather.get("description", ""), weather.get("humidity", 0)),
                    },
                }

            # Short human-friendly answer
            w = cleaned.get("weather", {})
            a = cleaned.get("aqi", {})
            analysis = cleaned.get("analysis", {})
            answer = (
                f"Thời tiết {city}: Nhiệt độ {w.get('temp')}°C, độ ẩm {w.get('humidity')}%, "
                f"{w.get('description')}. AQI {analysis.get('aqi')} ({analysis.get('aqi_category')}). "
                f"Khuyến nghị: {analysis.get('recommendation')}"
            )

            return {"success": True, "intent": "current_weather", "answer": answer, "cleaned": cleaned}

        # 2) Historical station stats (top / max / min)
        # Recognize metric names
        metric = payload.get("metric") or None
        if not metric:
            # heuristic mapping from text
            if "pm2" in q or "pm2.5" in q or "pm2,5" in q:
                metric = "PM2.5"
            elif "pm10" in q:
                metric = "PM10"
            elif "aqi" in q:
                metric = "AQI"

        # top N
        top_match = re.search(r"top\s*(\d+)|top-(\d+)|top(\d+)|([Tt]op)\s*(\d+)|cao nhất|lớn nhất|nhiều nhất", q)
        top_n = payload.get("top_n") or None
        if top_match and not top_n:
            # try to parse a number
            num = re.search(r"(\d+)", top_match.group(0) or "")
            if num:
                top_n = int(num.group(1))
        if top_n is None:
            top_n = int(payload.get("top_n") or 5)

        # top-date style query
        if any(k in q for k in ["top", "cao nhất", "lớn nhất", "nhiều nhất"]):
            df = _load_station_df()
            station_id = payload.get("station_id")
            if not metric:
                # default to PM2.5 if unknown
                metric = "PM2.5"
            try:
                records = _extract_top_n(df, metric, top_n=top_n, station_id=station_id)
            except KeyError as e:
                raise HTTPException(status_code=400, detail=str(e))

            answer = f"Top {top_n} ngày theo {metric}"
            if station_id:
                answer += f" cho trạm {station_id}"
            return {"success": True, "intent": "top_values", "answer": answer, "rows": records}

        # 3) Monthly trend / averages
        if any(k in q for k in ["trung bình tháng", "trung bình hàng tháng", "mỗi tháng", "trend", "xu hướng"]):
            df = _load_station_df()
            station_id = payload.get("station_id")
            if not metric:
                metric = "PM2.5"
            try:
                series = _monthly_average(df, metric, station_id=station_id)
            except KeyError as e:
                raise HTTPException(status_code=400, detail=str(e))
            answer = f"Trung bình {metric} theo tháng"
            return {"success": True, "intent": "monthly_trend", "answer": answer, "series": series}

        # 4) Fallback: return a polite message describing supported queries
        help_text = (
            "Tôi có thể trả lời các câu hỏi đơn giản sau: hiện tại (thời tiết, nhiệt độ, AQI),\n"
            "- Top N ngày theo PM2.5/PM10/AQI: ví dụ 'Top 5 ngày PM2.5 cao nhất'.\n"
            "- Trung bình theo tháng cho một metric: ví dụ 'Trung bình PM2.5 theo tháng'.\n"
            "Gửi 'question' cùng 'metric' hoặc 'station_id' tùy chọn trong payload."
        )
        return {"success": True, "intent": "help", "answer": help_text}

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
