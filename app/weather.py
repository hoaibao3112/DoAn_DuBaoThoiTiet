import os
import requests
import time
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, List
from app.gdrive import save_json_to_drive
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import ML predictor
try:
    from app.ml_predictor import ml_predictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML predictor not available. Run scripts/train_model.py to train models.")

# Load environment variables from .env at import-time so uvicorn processes pick them up
load_dotenv()

router = APIRouter(prefix="/weather", tags=["Weather & AQI"])


class WeatherRequest(BaseModel):
    city: str = "Hanoi"
    user_id: Optional[str] = None


class WeatherResponse(BaseModel):
    city: str
    date: str
    temperature: float
    feels_like: float
    humidity: int
    description: str
    aqi: int
    aqi_category: str
    pm25: float
    pm10: float
    recommendation: str
    source: str  # "api" or "ml_model"


class MLForecastRequest(BaseModel):
    """Request for ML-based forecast"""
    historical_data: List[dict]  # Past 7-30 days data
    days_ahead: int = 1  # Number of days to forecast


class MLForecastResponse(BaseModel):
    """Response with ML forecast"""
    success: bool
    forecast_date: Optional[str] = None
    pm25_forecast: Optional[float] = None
    aqi_forecast: Optional[int] = None
    aqi_category: Optional[str] = None
    confidence: Optional[str] = None
    recommendation: Optional[str] = None
    model_info: Optional[dict] = None
    error: Optional[str] = None


def get_weather_data(city: str) -> dict:
    """Get weather from OpenWeatherMap API"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key or api_key == "YOUR_OPENWEATHER_KEY":
        return {"error": "OpenWeather API key not configured"}
    
    base_url = os.getenv("OPENWEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5")
    url = f"{base_url}/weather"
    
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",  # Celsius
        "lang": "vi"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        return {
            "temp": round(data["main"]["temp"], 1),
            "feels_like": round(data["main"]["feels_like"], 1),
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "city_name": data["name"]
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Weather API error: {str(e)}"}
    except KeyError as e:
        return {"error": f"Invalid API response: {str(e)}"}


def get_aqi_data(city: str) -> dict:
    """Get AQI from World Air Quality Index (WAQI) API"""
    token = os.getenv("WAQI_TOKEN")
    if not token or token == "YOUR_WAQI_TOKEN":
        # Return mock data for testing
        return {
            "aqi": 85,
            "pm25": 35,
            "pm10": 60,
            "city_name": city,
            "source": "mock"
        }
    
    base_url = os.getenv("WAQI_BASE_URL", "https://api.waqi.info")
    url = f"{base_url}/feed/{city.lower()}/"
    
    params = {"token": token}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("status") != "ok":
            return {"error": f"WAQI API error: {data.get('data', 'Unknown error')}"}
        
        aqi_data = data["data"]
        iaqi = aqi_data.get("iaqi", {})
        
        return {
            "aqi": aqi_data.get("aqi", 0),
            "pm25": iaqi.get("pm25", {}).get("v", 0),
            "pm10": iaqi.get("pm10", {}).get("v", 0),
            "city_name": aqi_data.get("city", {}).get("name", city),
            "source": "waqi_api"
        }
    except requests.exceptions.RequestException as e:
        # Fallback to mock data
        return {
            "aqi": 85,
            "pm25": 35,
            "pm10": 60,
            "city_name": city,
            "source": "fallback"
        }


def get_aqi_category(aqi: int) -> str:
    """Convert AQI number to category"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def fetch_raw_weather_and_aqi(city: str) -> dict:
    """Fetch raw JSON responses from OpenWeather and WAQI (when available).
    Returns a dict with keys 'weather_raw' and 'aqi_raw' (may contain 'error').
    """
    result = {"weather_raw": None, "aqi_raw": None}

    # OpenWeather raw
    api_key = os.getenv("OPENWEATHER_API_KEY")
    base_url = os.getenv("OPENWEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5")
    if api_key and api_key != "YOUR_OPENWEATHER_KEY":
        try:
            resp = requests.get(f"{base_url}/weather", params={"q": city, "appid": api_key, "units": "metric"}, timeout=10)
            resp.raise_for_status()
            result["weather_raw"] = resp.json()
        except Exception as e:
            result["weather_raw"] = {"error": str(e)}
    else:
        result["weather_raw"] = {"error": "OpenWeather API key not configured"}

    # WAQI raw
    token = os.getenv("WAQI_TOKEN")
    waqi_base = os.getenv("WAQI_BASE_URL", "https://api.waqi.info")
    if token and token != "YOUR_WAQI_TOKEN":
        try:
            resp = requests.get(f"{waqi_base}/feed/{city.lower()}/", params={"token": token}, timeout=10)
            resp.raise_for_status()
            result["aqi_raw"] = resp.json()
        except Exception as e:
            result["aqi_raw"] = {"error": str(e)}
    else:
        result["aqi_raw"] = {"error": "WAQI token not configured"}

    return result


def fetch_openweather_air_by_coord(lat: float, lon: float) -> dict:
    """Fetch OpenWeather Air Pollution components for given coordinates.
    Returns dict like {'components': {...}, 'dt': unix_timestamp} or {'error': ...}
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key or api_key == "YOUR_OPENWEATHER_KEY":
        return {"error": "OpenWeather API key not configured"}

    base_url = os.getenv("OPENWEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5")
    try:
        resp = requests.get(f"{base_url}/air_pollution", params={"lat": lat, "lon": lon, "appid": api_key}, timeout=8)
        resp.raise_for_status()
        j = resp.json()
        if isinstance(j, dict) and j.get('list'):
            entry = j['list'][0]
            return {"components": entry.get('components', {}), "dt": entry.get('dt')}
        return {"error": "No air pollution data returned"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/ow-forecast")
def openweather_onecall_forecast(city: str = "Hanoi", date: str = None):
    """Server-side helper: geocode `city` then call OpenWeather One Call to get daily forecast for `date`.
    Returns JSON: {"success": True, "summary": "..."} or {"success": False, "error": "..."}
    `date` should be YYYY-MM-DD (optional, defaults to tomorrow).
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenWeather API key not configured")

    base_url = os.getenv("OPENWEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5").rstrip('/')

    # determine target date
    try:
        if date:
            tgt = datetime.fromisoformat(date).date()
        else:
            from datetime import date as _date, timedelta
            tgt = (_date.today() + timedelta(days=1))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format; use YYYY-MM-DD")

    # geocode city -> lat/lon using OpenWeather Geocoding API
    geocode_url = f"https://api.openweathermap.org/geo/1.0/direct"
    try:
        gresp = requests.get(geocode_url, params={"q": city, "limit": 1, "appid": api_key}, timeout=6)
        gresp.raise_for_status()
        gjson = gresp.json()
        if not isinstance(gjson, list) or len(gjson) == 0:
            raise HTTPException(status_code=404, detail=f"Geocoding failed for city={city}")
        lat = gjson[0].get('lat')
        lon = gjson[0].get('lon')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Geocoding error: {e}")

    # call One Call for daily forecasts
    onecall_url = f"{base_url}/onecall"
    try:
        resp = requests.get(onecall_url, params={"lat": lat, "lon": lon, "exclude": "minutely,hourly,alerts", "units": "metric", "appid": api_key}, timeout=8)
        resp.raise_for_status()
        j = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenWeather One Call error: {e}")

    # find matching daily entry
    daily = j.get('daily') or []
    match = None
    for entry in daily:
        try:
            entry_dt = datetime.utcfromtimestamp(entry.get('dt')).date()
            if entry_dt == tgt:
                match = entry
                break
        except Exception:
            continue

    if not match:
        return {"success": False, "error": "No daily forecast for requested date in One Call response"}

    # summarize and prepare detailed match info
    temp_obj = match.get('temp', {})
    day_temp = temp_obj.get('day')
    min_temp = temp_obj.get('min')
    max_temp = temp_obj.get('max')
    pop = match.get('pop')  # probability of precipitation
    weather = match.get('weather', [])
    desc = weather[0].get('description') if weather and isinstance(weather, list) else ''

    parts = []
    if day_temp is not None:
        parts.append(f"OpenWeather: Nhiệt độ trung bình dự báo ~{day_temp:.1f}°C (min {min_temp:.1f}°C / max {max_temp:.1f}°C)")
    if pop is not None:
        parts.append(f"Khả năng mưa: ~{int(pop*100)}%")
    if desc:
        parts.append(f"Mô tả: {desc}")

    summary = "; ".join(parts) if parts else None

    # Build a compact detailed dict to help frontend debugging (avoid extremely large payload)
    detail = {
        'geocode': {'lat': lat, 'lon': lon, 'resolved_name': gjson[0].get('name') if isinstance(gjson, list) and gjson else None},
        'daily_match': {
            'dt': match.get('dt'),
            'temp': {
                'day': day_temp,
                'min': min_temp,
                'max': max_temp
            },
            'pop': pop,
            'weather': weather[0] if isinstance(weather, list) and weather else None
        }
    }

    return {"success": True, "summary": summary, "detail": detail}


def clean_and_summarize_raw(city: str, raw: dict) -> dict:
    """Extract cleaned fields and summary from raw responses."""
    weather_raw = raw.get("weather_raw") or {}
    aqi_raw = raw.get("aqi_raw") or {}

    cleaned = {
        "city": city,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "weather": {},
        "aqi": {},
        "analysis": {}
    }

    # Clean weather
    if isinstance(weather_raw, dict) and "error" not in weather_raw:
        try:
            cleaned["weather"] = {
                "temp": weather_raw.get("main", {}).get("temp"),
                "feels_like": weather_raw.get("main", {}).get("feels_like"),
                "humidity": weather_raw.get("main", {}).get("humidity"),
                "description": weather_raw.get("weather", [{}])[0].get("description"),
                "wind_speed": weather_raw.get("wind", {}).get("speed"),
                "raw": weather_raw
            }
        except Exception:
            cleaned["weather"] = {"raw": weather_raw}
    else:
        cleaned["weather"] = {"error": weather_raw.get("error") if isinstance(weather_raw, dict) else str(weather_raw)}

    # Normalize coordinates to top-level `cleaned['coord']` for easier frontend usage
    try:
        coord_val = None
        if isinstance(weather_raw, dict) and weather_raw.get('coord'):
            coord_val = weather_raw.get('coord')
        # sometimes coord is nested under cleaned['weather']['raw']
        if not coord_val:
            w_raw = cleaned.get('weather', {}).get('raw')
            if isinstance(w_raw, dict) and w_raw.get('coord'):
                coord_val = w_raw.get('coord')
        # fallback: raw may contain lat/lon keys
        if not coord_val and isinstance(weather_raw, dict) and 'lat' in weather_raw and 'lon' in weather_raw:
            coord_val = {'lat': weather_raw.get('lat'), 'lon': weather_raw.get('lon')}

        if coord_val and (coord_val.get('lat') is not None or coord_val.get('latitude') is not None):
            lat = coord_val.get('lat') or coord_val.get('latitude')
            lon = coord_val.get('lon') or coord_val.get('longitude') or coord_val.get('lng')
            cleaned['coord'] = {'lat': lat, 'lon': lon}
    except Exception:
        # don't fail if coordinate normalization fails
        pass

    # Clean aqi
    if isinstance(aqi_raw, dict) and aqi_raw.get("status") == "ok":
        try:
            data = aqi_raw.get("data", {})
            iaqi = data.get("iaqi", {})
            cleaned["aqi"] = {
                "aqi": data.get("aqi"),
                "pm25": iaqi.get("pm25", {}).get("v") if iaqi.get("pm25") else None,
                "pm10": iaqi.get("pm10", {}).get("v") if iaqi.get("pm10") else None,
                "raw": aqi_raw
            }
        except Exception:
            cleaned["aqi"] = {"raw": aqi_raw}
    else:
        # If WAQI returned error or token missing, try a fallback strategy:
        # 1) If aqi_raw contains an explicit error, log it and try OpenWeather air by coord
        # 2) Use get_aqi_data() as a second attempt (it may return mock/fallback)
        # 3) If still missing, use a safe mock fallback so downstream code gets numeric values
        if isinstance(aqi_raw, dict) and aqi_raw.get("error"):
            print(f"WAQI fetch error for {city}: {aqi_raw.get('error')}")

        # First try the helper which may return real values or an error dict
        a = get_aqi_data(city)

        # If helper returned error OR returned None for key pollutant values, attempt coordinate fallback
        need_coord_fallback = False
        if not isinstance(a, dict):
            need_coord_fallback = True
        else:
            if a.get("error"):
                need_coord_fallback = True
            # If numeric pollutant values are missing, try OpenWeather air_pollution
            if a.get("aqi") is None or a.get("pm25") is None or a.get("pm10") is None:
                need_coord_fallback = True

        if need_coord_fallback:
            coord = weather_raw.get('coord') if isinstance(weather_raw, dict) else None
            if coord and coord.get('lat') and coord.get('lon'):
                print(f"Attempting OpenWeather air_pollution fallback for {city} at {coord}")
                air = fetch_openweather_air_by_coord(coord.get('lat'), coord.get('lon'))
                if isinstance(air, dict) and air.get('components'):
                    comps = air.get('components')
                    cleaned['aqi'] = {
                        'aqi': a.get('aqi') if isinstance(a, dict) and a.get('aqi') is not None else None,
                        'pm25': comps.get('pm2_5'),
                        'pm10': comps.get('pm10'),
                        'source': (a.get('source') + ';' if isinstance(a, dict) and a.get('source') else '') + 'openweather'
                    }
                else:
                    print(f"OpenWeather air_pollution fallback failed for {city}: {air}")
                    cleaned['aqi'] = {
                        'aqi': a.get('aqi') if isinstance(a, dict) and a.get('aqi') is not None else 85,
                        'pm25': a.get('pm25') if isinstance(a, dict) and a.get('pm25') is not None else 35,
                        'pm10': a.get('pm10') if isinstance(a, dict) and a.get('pm10') is not None else 60,
                        'source': a.get('source') if isinstance(a, dict) and a.get('source') else 'fallback'
                    }
            else:
                # no coordinates available - use whatever helper provided or fallback mock
                cleaned['aqi'] = {
                    'aqi': a.get('aqi') if isinstance(a, dict) and a.get('aqi') is not None else 85,
                    'pm25': a.get('pm25') if isinstance(a, dict) and a.get('pm25') is not None else 35,
                    'pm10': a.get('pm10') if isinstance(a, dict) and a.get('pm10') is not None else 60,
                    'source': a.get('source') if isinstance(a, dict) and a.get('source') else 'fallback'
                }
        else:
            # use the values returned from get_aqi_data (may be mock or real)
            cleaned["aqi"] = {
                "aqi": a.get("aqi"),
                "pm25": a.get("pm25"),
                "pm10": a.get("pm10"),
                "source": a.get("source")
            }

    # If we have weather_raw with coordinates, try to fetch OpenWeather air components and merge
    try:
        weather_raw = raw.get("weather_raw") or {}
        coord = weather_raw.get('coord') if isinstance(weather_raw, dict) else None
        if coord and coord.get('lat') and coord.get('lon'):
            air = fetch_openweather_air_by_coord(coord.get('lat'), coord.get('lon'))
            if isinstance(air, dict) and air.get('components'):
                comps = air.get('components')
                # map OpenWeather components (pm2_5 -> pm25)
                if comps.get('pm2_5') is not None:
                    cleaned['aqi']['pm25'] = comps.get('pm2_5')
                if comps.get('pm10') is not None:
                    cleaned['aqi']['pm10'] = comps.get('pm10')
                # also include other gases
                for gas in ['no2', 'so2', 'o3', 'co']:
                    if comps.get(gas) is not None:
                        cleaned['aqi'][gas] = comps.get(gas)
                cleaned['aqi']['source'] = cleaned['aqi'].get('source', '') + ';openweather' if cleaned['aqi'].get('source') else 'openweather'
    except Exception:
        pass

    # Analysis: category & recommendation
    aqi_val = cleaned["aqi"].get("aqi") if isinstance(cleaned["aqi"], dict) else None
    temp = cleaned["weather"].get("temp") if isinstance(cleaned["weather"], dict) else None
    desc = cleaned["weather"].get("description") if isinstance(cleaned["weather"], dict) else ""
    humidity = cleaned["weather"].get("humidity") if isinstance(cleaned["weather"], dict) else None

    if aqi_val is not None:
        try:
            aqi_int = int(aqi_val)
            category = get_aqi_category(aqi_int)
        except Exception:
            aqi_int = None
            category = None
    else:
        aqi_int = None
        category = None

    # Pass None for missing AQI so recommendation isn't based on a fabricated 0 value
    cleaned["analysis"] = {
        "aqi": aqi_int,
        "aqi_category": category,
        "recommendation": generate_recommendation(aqi_int, temp, desc or "", humidity)
    }

    return cleaned


def generate_recommendation(aqi: Optional[int], temp: Optional[float], description: str, humidity: Optional[int]) -> str:
    """Generate personalized health and activity recommendations.

    If AQI is None, do not fabricate a 'Good' message; instead note AQI missing
    and provide weather-based suggestions only.
    """
    recs = []

    # AQI-based recommendations
    if aqi is None:
        recs.append("ℹ️ Dữ liệu AQI hiện chưa có — khuyến nghị chỉ dựa trên thời tiết.")
    else:
        try:
            if aqi <= 50:
                recs.append("🌟 Chất lượng không khí tốt! Thích hợp cho mọi hoạt động ngoài trời.")
            elif aqi <= 100:
                recs.append("🟡 Không khí ở mức trung bình. Người nhạy cảm nên hạn chế hoạt động ngoài trời kéo dài.")
            elif aqi <= 150:
                recs.append("🟠 Không khí không lành mạnh cho nhóm nhạy cảm (trẻ em, người già, bệnh hô hấp). Nên đeo khẩu trang N95 khi ra ngoài.")
            elif aqi <= 200:
                recs.append("🔴 Không khí ô nhiễm! Mọi người nên hạn chế ra ngoài. Đóng cửa sổ, bật máy lọc không khí.")
            else:
                recs.append("⚠️ CẢNH BÁO: Không khí cực kỳ ô nhiễm! Ở trong nhà, đóng kín cửa, sử dụng máy lọc không khí. Tránh mọi hoạt động ngoài trời.")
        except Exception:
            recs.append("ℹ️ Không có dữ liệu AQI hợp lệ để đưa ra khuyến nghị.")

    # Weather-based recommendations
    try:
        if description and ("rain" in description.lower() or "mưa" in description.lower()):
            recs.append("☔ Có mưa - nhớ mang theo ô hoặc áo mưa.")
    except Exception:
        pass

    if temp is not None:
        try:
            if temp > 35:
                recs.append("🌡️ Nhiệt độ rất cao - uống nhiều nước, tránh tiếp xúc trực tiếp với nắng từ 11h-15h.")
            elif temp > 30:
                recs.append("☀️ Trời nắng nóng - mặc quần áo thoáng mát, dùng kem chống nắng.")
            elif temp < 15:
                recs.append("🧥 Trời lạnh - mặc ấm khi ra ngoài, đặc biệt vào buổi sáng sớm và tối.")
            elif temp < 10:
                recs.append("❄️ Trời rất lạnh - mặc nhiều lớp áo, đội mũ, đeo khăn để giữ ấm.")
        except Exception:
            pass

    if humidity is not None:
        try:
            if humidity > 80:
                recs.append("💧 Độ ẩm cao - có thể cảm thấy oi bức và khó chịu.")
        except Exception:
            pass

    return " ".join(recs)


@router.get("/current")
def get_current_weather(city: str = "Hanoi"):
    """Get current weather and AQI for a city"""
    # Get weather data
    weather = get_weather_data(city)
    if "error" in weather:
        raise HTTPException(status_code=500, detail=weather["error"])
    
    # Get AQI data
    aqi_data = get_aqi_data(city)
    if "error" in aqi_data:
        # Use fallback values
        aqi_data = {"aqi": 100, "pm25": 50, "pm10": 80, "source": "fallback"}
    
    aqi = int(aqi_data["aqi"])
    category = get_aqi_category(aqi)
    recommendation = generate_recommendation(
        aqi, 
        weather["temp"], 
        weather["description"],
        weather["humidity"]
    )
    
    return WeatherResponse(
        city=weather.get("city_name", city),
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        temperature=weather["temp"],
        feels_like=weather["feels_like"],
        humidity=weather["humidity"],
        description=weather["description"],
        aqi=aqi,
        aqi_category=category,
        pm25=float(aqi_data.get("pm25", 0)),
        pm10=float(aqi_data.get("pm10", 0)),
        recommendation=recommendation,
        source=aqi_data.get("source", "api")
    )


@router.post("/forecast", response_model=WeatherResponse)
def get_forecast(request: WeatherRequest):
    """Get weather forecast with AQI (chatbot-friendly endpoint)"""
    return get_current_weather(request.city)


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "apis": {
            "openweather": os.getenv("OPENWEATHER_API_KEY", "").startswith("YOUR") == False,
            "waqi": os.getenv("WAQI_TOKEN", "").startswith("YOUR") == False
        }
    }


# Simple in-memory cache for expensive neighbor computations
_NEIGHBOR_CACHE = {}
# Optional Redis client (if REDIS_URL provided)
_REDIS_CLIENT = None
try:
    REDIS_URL = os.getenv('REDIS_URL')
    if REDIS_URL:
        try:
            import redis
            _REDIS_CLIENT = redis.from_url(REDIS_URL)
        except Exception:
            _REDIS_CLIENT = None
except Exception:
    _REDIS_CLIENT = None


def _cache_set(key: str, value: dict, ttl: int = 600):
    if _REDIS_CLIENT:
        try:
            _REDIS_CLIENT.set(key, json.dumps(value), ex=ttl)
            return
        except Exception:
            pass
    _NEIGHBOR_CACHE[key] = (time.time() + ttl, value)


def _cache_get(key: str):
    # Try Redis first
    if _REDIS_CLIENT:
        try:
            raw = _REDIS_CLIENT.get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            pass

    v = _NEIGHBOR_CACHE.get(key)
    if not v:
        return None
    expire, val = v
    if time.time() > expire:
        try:
            del _NEIGHBOR_CACHE[key]
        except Exception:
            pass
        return None
    return val


@router.get("/neighbor-prob")
def neighbor_probability(station: str, csv_path: str = None, metric: str = "pm25", min_overlap: int = 120, top_k: int = 3, precip_threshold: float = 0.5):
    """Compute weighted neighbor-based rain probability for `station` using historical CSV.

    - `station`: StationId / station name that matches CSV index
    - `csv_path`: path to CSV (defaults to `data/cleaned/station_day_clean.csv`)
    - `metric`: which pollutant to use for correlation ('pm25' or 'pm10')
    - `min_overlap`: minimum days of overlap to consider correlation valid
    - `top_k`: number of neighbors to use
    Returns JSON: {success: True, prob: 0.32, neighbors: [...], used_metric: 'pm25'}
    """
    key = f"neighbor:{station}:{metric}:{min_overlap}:{top_k}:{precip_threshold}"
    cached = _cache_get(key)
    if cached:
        return {"success": True, "cached": True, **cached}

    csv_path = csv_path or os.path.join(os.getcwd(), 'data', 'cleaned', 'station_day_clean.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {e}")

    # Normalize column names to lower-case keys
    colmap = {c.lower().strip(): c for c in df.columns}
    # Identify station id col and date col
    station_col = None
    date_col = None
    for token in ('stationid','station','station_id','station code'):
        if token in colmap:
            station_col = colmap[token]
            break
    for token in ('date','day','timestamp'):
        if token in colmap:
            date_col = colmap[token]
            break
    if not station_col or not date_col:
        raise HTTPException(status_code=400, detail="Could not find station/date columns in CSV")

    # find precipitation column or rain flag
    precip_col = None
    rainflag_col = None
    for token in ('precip_mm','precipitation','rain_mm','rain','precip'):
        if token in colmap:
            precip_col = colmap[token]
            break
    for token in ('rain_flag','rainflag','rflag'):
        if token in colmap:
            rainflag_col = colmap[token]
            break

    # candidate metric column names
    metric_tokens = [metric.lower(), 'pm2.5', 'pm25', 'pm10']
    metric_col = None
    for t in metric_tokens:
        if t in colmap:
            metric_col = colmap[t]
            break
    if metric_col is None:
        # fallback choose any of pm25/pm10
        for t in ('pm25','pm10'):
            if t in colmap:
                metric_col = colmap[t]
                break

    # pivot to station x date
    try:
        sub = df[[station_col, date_col, metric_col]].copy() if metric_col in df.columns else None
        sub = df[[station_col, date_col]].copy().assign(metric=0) if sub is None else sub
        if metric_col and metric_col in sub.columns:
            sub = sub.rename(columns={metric_col: 'metric'})
        else:
            sub = sub.rename(columns={sub.columns[-1]: 'metric'})

        # ensure date type
        sub[date_col] = pd.to_datetime(sub[date_col], errors='coerce')
        sub = sub.dropna(subset=[date_col])
        sub['Date'] = sub[date_col].dt.strftime('%Y-%m-%d')
        pivot = sub.pivot_table(index='Date', columns=station_col, values='metric')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pivot error: {e}")

    if station not in pivot.columns:
        raise HTTPException(status_code=404, detail=f"Station {station} not found in CSV")

    target = pivot[station]
    corrs = pivot.corrwith(target, axis=0)
    # compute valid overlap counts
    overlaps = pivot.notna() & target.notna()
    overlap_counts = overlaps.sum()

    # filter neighbors
    neighbors = []
    for other, corr in corrs.drop(labels=[station], errors='ignore').dropna().items():
        ov = int(overlap_counts.get(other, 0))
        if ov >= min_overlap:
            neighbors.append({'station': other, 'corr': float(corr), 'overlap': int(ov)})

    if not neighbors:
        # no neighbors -> cannot estimate
        result = {"prob": None, "neighbors": [], "used_metric": metric_col}
        _cache_set(key, result)
        return {"success": True, **result}

    # choose top_k by absolute correlation
    neighbors = sorted(neighbors, key=lambda x: abs(x['corr']), reverse=True)[:top_k]

    # compute each neighbor's historic rain fraction
    details = []
    weights = []
    probs = []
    for n in neighbors:
        other = n['station']
        # build mask of days where both have data
        both = pivot[[station, other]].dropna()
        rain_vals = None
        if rainflag_col and rainflag_col in df.columns:
            # compute fraction for that neighbor from original df
            neigh_df = df[df[station_col] == other]
            if not neigh_df.empty:
                rf = neigh_df[rainflag_col].dropna()
                try:
                    frac = float((rf > 0).sum()) / max(1, len(rf))
                except Exception:
                    frac = None
            else:
                frac = None
        elif precip_col and precip_col in df.columns:
            neigh_df = df[df[station_col] == other]
            if not neigh_df.empty:
                pvals = pd.to_numeric(neigh_df[precip_col], errors='coerce').dropna()
                try:
                    frac = float((pvals > precip_threshold).sum()) / max(1, len(pvals))
                except Exception:
                    frac = None
            else:
                frac = None
        else:
            frac = None

        details.append({**n, 'rain_frac': frac})
        w = abs(n['corr']) if n['corr'] is not None else 0.0
        weights.append(w)
        probs.append(frac if frac is not None else 0.0)

    # weighted probability
    weights = np.array(weights, dtype=float)
    probs = np.array(probs, dtype=float)
    if weights.sum() <= 0:
        prob = float(np.nan)
    else:
        prob = float(np.sum(weights * probs) / weights.sum())

    result = {"prob": prob, "neighbors": details, "used_metric": metric_col}
    _cache_set(key, result)
    return {"success": True, **result}


@router.post("/save-clean")
def save_cleaned_weather(city: str = "Ho Chi Minh", folder_id: Optional[str] = None):
    """Fetch raw weather & AQI for `city`, clean and analyze them, and save cleaned JSON to Google Drive (if folder_id configured).
    Returns the cleaned record and Drive file id when saved.
    """
    raw = fetch_raw_weather_and_aqi(city)
    cleaned = clean_and_summarize_raw(city, raw)

    # Append cleaned numeric pollutants to local CSV for historical analysis
    try:
        append_cleaned_to_csv(cleaned)
    except Exception as e:
        print("Failed to append cleaned record to CSV:", e)

    file_id = None
    target_folder = folder_id or os.getenv("GDRIVE_FOLDER_ID")
    if target_folder:
        try:
            file_id = save_json_to_drive(cleaned, target_folder)
        except Exception as e:
            # Log and continue
            print("Drive save failed:", e)

    return {"success": True, "cleaned": cleaned, "drive_file_id": file_id}


def append_cleaned_to_csv(cleaned: dict, csv_path: str = None):
    """Append a normalized row to data/cleaned/station_day_clean.csv.
    The function will try to map common pollutant fields into the CSV schema.
    """
    import pandas as pd
    from datetime import datetime

    csv_path = csv_path or os.path.join(os.getcwd(), 'data', 'cleaned', 'station_day_clean.csv')
    # ensure directory exists
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Prepare row dict with common columns seen in existing CSV
    row = {}
    # Basic mapping
    ts = cleaned.get('timestamp_utc') or datetime.utcnow().isoformat() + 'Z'
    try:
        dt = datetime.fromisoformat(ts.replace('Z', ''))
        row['Date'] = dt.strftime('%Y-%m-%d')
    except Exception:
        row['Date'] = ts

    # Pollutants
    aqi = cleaned.get('analysis', {}).get('aqi') or cleaned.get('aqi', {}).get('aqi')
    row['AQI'] = aqi
    # PM2.5
    pm25 = cleaned.get('aqi', {}).get('pm25')
    if pm25 is None:
        pm25 = cleaned.get('aqi', {}).get('PM2.5')
    row['PM2.5'] = pm25
    # PM10
    pm10 = cleaned.get('aqi', {}).get('pm10')
    row['PM10'] = pm10

    # Precipitation (try multiple common locations in cleaned/raw payload)
    precip_mm = None
    try:
        # Prefer cleaned['weather']['raw'] (OpenWeather current) which may include 'rain' dict
        wraw = (cleaned.get('weather') or {}).get('raw') if isinstance(cleaned.get('weather'), dict) else None
        if not wraw:
            # fallback to top-level raw
            wraw = cleaned.get('raw') or {}
        if isinstance(wraw, dict):
            # OpenWeather current: 'rain': {'1h': val} or {'3h': val}
            rain = wraw.get('rain')
            if isinstance(rain, dict):
                precip_mm = rain.get('1h') or rain.get('3h') or precip_mm
            elif isinstance(rain, (int, float)):
                precip_mm = float(rain)
            # Some responses include 'precipitation' or onecall daily 'rain'
            if precip_mm is None:
                if 'precipitation' in wraw:
                    try:
                        precip_mm = float(wraw.get('precipitation'))
                    except Exception:
                        precip_mm = precip_mm
                else:
                    try:
                        # try nested daily_match if present (from ow-forecast detail attached earlier)
                        detail = cleaned.get('detail') or {}
                        dm = (detail.get('daily_match') or {})
                        if dm and dm.get('rain') is not None:
                            precip_mm = float(dm.get('rain'))
                    except Exception:
                        pass
        # final fallback: direct keys
        if precip_mm is None:
            for k in ('precip_mm', 'rain_mm', 'rain'):
                v = cleaned.get(k)
                if v is not None:
                    try:
                        precip_mm = float(v)
                        break
                    except Exception:
                        continue
    except Exception:
        precip_mm = None

    # Save precip and a simple rain flag (1 if >0.5 mm else 0)
    row['Precip_mm'] = precip_mm
    try:
        if precip_mm is None:
            row['Rain_flag'] = None
        else:
            row['Rain_flag'] = 1 if float(precip_mm) > 0.5 else 0
    except Exception:
        row['Rain_flag'] = None

    # other gases
    for gas_col in [('NO2', 'no2'), ('SO2', 'so2'), ('O3', 'o3'), ('CO', 'co')]:
        colname, key = gas_col
        val = cleaned.get('aqi', {}).get(key)
        if val is not None:
            row[colname] = val

    # Fill additional fields with None/defaults to match schema flexibility
    # If file exists, respect its columns; otherwise write minimal set
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Ensure all columns present in df
        for c in df.columns:
            if c not in row:
                row[c] = None
        # Append row and save
        df = df.append(row, ignore_index=True)
        df.to_csv(csv_path, index=False)
    else:
        # Create new DataFrame with this single row
        df = pd.DataFrame([row])
        df.to_csv(csv_path, index=False)


@router.post("/forecast-ml", response_model=MLForecastResponse)
def forecast_with_ml(request: MLForecastRequest):
    """
    Get ML-based weather forecast using trained RandomForest models
    
    Requires historical data (past 7-30 days) with format:
    [
        {'date': '2024-01-01', 'pm25': 45.2, 'pm10': 89.1, 'aqi': 102, ...},
        {'date': '2024-01-02', 'pm25': 50.3, 'pm10': 95.2, 'aqi': 115, ...},
        ...
    ]
    """
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML models not available. Train models first with: python scripts/train_model.py"
        )
    
    if not ml_predictor.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Run scripts/train_model.py to train models."
        )
    
    # Validate input
    if len(request.historical_data) < 7:
        raise HTTPException(
            status_code=400,
            detail="Need at least 7 days of historical data for accurate predictions"
        )
    
    # Make prediction
    if request.days_ahead == 1:
        result = ml_predictor.predict_next_day(request.historical_data)
    else:
        # Multi-day forecast
        results = ml_predictor.batch_predict(request.historical_data, days_ahead=request.days_ahead)
        if not results:
            result = {'success': False, 'error': 'Batch prediction failed'}
        else:
            result = results[-1]  # Return last prediction
    
    if not result['success']:
        raise HTTPException(status_code=500, detail=result.get('error', 'Prediction failed'))
    
    # Generate category and recommendation
    aqi_category = get_aqi_category(result['aqi_forecast'])
    
    # Use mock weather data for recommendation (or integrate with real API)
    recommendation = generate_recommendation(
        aqi=result['aqi_forecast'],
        temp=25.0,  # Default temp, can be enhanced with weather API
        description="Partly cloudy",
        humidity=70
    )
    
    return MLForecastResponse(
        success=True,
        forecast_date=result['forecast_date'],
        pm25_forecast=result['pm25_forecast'],
        aqi_forecast=result['aqi_forecast'],
        aqi_category=aqi_category,
        confidence=result['confidence'],
        recommendation=recommendation,
        model_info=result['model_info']
    )


@router.get("/forecast-ml/batch/{days}")
def forecast_batch_ml(days: int = 7):
    """
    Get multi-day ML forecast (demo endpoint with mock historical data)
    
    In production, this would fetch real historical data from database
    """
    if not ML_AVAILABLE or not ml_predictor.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="ML models not available"
        )
    
    # Mock historical data (in production, fetch from database)
    from datetime import timedelta
    today = datetime.now()
    historical_data = []
    
    for i in range(30, 0, -1):
        date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        historical_data.append({
            'date': date,
            'pm25': 45.0 + i * 0.5,  # Mock data
            'pm10': 80.0 + i * 0.8,
            'aqi': int(100 + i * 1.2),
            'no': 10 + i * 0.1,
            'no2': 30 + i * 0.2
        })
    
    # Get batch predictions
    results = ml_predictor.batch_predict(historical_data, days_ahead=days)
    
    # Format response
    forecasts = []
    for result in results:
        forecasts.append({
            'date': result['forecast_date'],
            'pm25': result['pm25_forecast'],
            'aqi': result['aqi_forecast'],
            'category': get_aqi_category(result['aqi_forecast']),
            'confidence': result['confidence']
        })
    
    return {
        'success': True,
        'forecasts': forecasts,
        'model_info': results[0]['model_info'] if results else {}
    }
