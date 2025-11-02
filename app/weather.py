import os
import requests
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import ML predictor
try:
    from app.ml_predictor import ml_predictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML predictor not available. Run scripts/train_model.py to train models.")

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


def generate_recommendation(aqi: int, temp: float, description: str, humidity: int) -> str:
    """Generate personalized health and activity recommendations"""
    recs = []
    
    # AQI-based recommendations
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
    
    # Weather-based recommendations
    if "rain" in description.lower() or "mưa" in description.lower():
        recs.append("☔ Có mưa - nhớ mang theo ô hoặc áo mưa.")
    
    if temp > 35:
        recs.append("🌡️ Nhiệt độ rất cao - uống nhiều nước, tránh tiếp xúc trực tiếp với nắng từ 11h-15h.")
    elif temp > 30:
        recs.append("☀️ Trời nắng nóng - mặc quần áo thoáng mát, dùng kem chống nắng.")
    elif temp < 15:
        recs.append("🧥 Trời lạnh - mặc ấm khi ra ngoài, đặc biệt vào buổi sáng sớm và tối.")
    elif temp < 10:
        recs.append("❄️ Trời rất lạnh - mặc nhiều lớp áo, đội mũ, đeo khăn để giữ ấm.")
    
    if humidity > 80:
        recs.append("💧 Độ ẩm cao - có thể cảm thấy oi bức và khó chịu.")
    
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
