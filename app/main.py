import os
import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException

from app.schemas import GenerateRequest, GenerateResponse
from app.gdrive import save_json_to_drive
from app.zalo import send_message_to_zalo
from app.weather import router as weather_router
from app.weather import get_weather_data, get_aqi_data, get_aqi_category, generate_recommendation
# Import neighbor helper to expose a stable route in case router wasn't updated
try:
    from app.weather import neighbor_probability
except Exception:
    neighbor_probability = None
from app.nlq import router as nlq_router

try:
    import openai
except Exception:
    openai = None

APP_PORT = int(os.getenv("APP_PORT", "8000"))

app = FastAPI(title="AI Zalo Assistant + Weather Forecast")

# Include weather router
app.include_router(weather_router)
# Include NLQ router
app.include_router(nlq_router)


def generate_ad_text(prompt: str) -> str:
    """Generate ad text. If OPENAI_API_KEY is configured use OpenAI, else fallback to a simple template."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and openai is not None:
        openai.api_key = api_key
        try:
            resp = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý viết quảng cáo ngắn gọn, bằng tiếng Việt."},
                    {"role": "user", "content": f"Viết một đoạn quảng cáo ngắn cho: {prompt}"},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            text = resp["choices"][0]["message"]["content"].strip()
            return text
        except Exception as e:
            # fallback to template on error
            print("OpenAI generation failed:", e)

    # Fallback simple template
    return f"Khám phá sản phẩm: {prompt}. Liên hệ ngay để biết thêm chi tiết và nhận ưu đãi!"


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest):
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # 1) generate ad text (synchronously so we can reply)
    ad_text = generate_ad_text(prompt)

    # 2) prepare JSON record
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prompt": prompt,
        "response": ad_text,
    }

    # 3) Save to Google Drive (synchronously). If drive upload fails, we still return the ad_text.
    drive_file_id = None
    folder_id = os.getenv("GDRIVE_FOLDER_ID")
    try:
        if folder_id:
            drive_file_id = save_json_to_drive(record, folder_id)
    except Exception as e:
        print("Google Drive upload failed:", e)

    # 4) Send response to Zalo user (synchronously so user receives reply right away)
    try:
        if payload.user_id:
            send_result = send_message_to_zalo(payload.user_id, ad_text)
            # optionally inspect send_result for errors
    except Exception as e:
        print("Zalo send failed:", e)

    return GenerateResponse(success=True, ad_text=ad_text, drive_file_id=drive_file_id)


@app.post("/ask")
def ask_telegram_update(payload: dict):
    """Handle a Telegram-like update payload coming from n8n.
    - If the message mentions weather ("thời tiết"), call the weather helper and return a friendly summary.
    - Otherwise forward the text to the generator as a fallback.
    The interaction is saved to Google Drive when `GDRIVE_FOLDER_ID` is configured.
    """
    try:
        # Normalize incoming payload - accept either full Telegram update or a body wrapper
        body = payload.get('body') if isinstance(payload, dict) and payload.get('body') else payload
        message = body.get('message') if isinstance(body, dict) else None
        text = ''
        chat_id = None
        if message:
            text = message.get('text', '') or ''
            chat = message.get('chat', {})
            chat_id = chat.get('id')
        else:
            # Fallback: try common keys
            text = body.get('text', '') if isinstance(body, dict) else ''

        text = (text or '').strip()

        # Decide intent
        lower = text.lower()
        reply_text = ''
        used_city = 'Ho Chi Minh'
        if 'thời tiết' in lower or 'thoi tiet' in lower or 'hcm' in lower or 'hồ chí minh' in lower:
            # Use weather helpers
            city = 'Ho Chi Minh'
            weather = get_weather_data(city)
            if 'error' in weather:
                # Return a friendly reply instead of raising an HTTP error so UI doesn't get 502
                err = weather.get('error')
                reply_text = (
                    f"Không thể lấy dữ liệu thời tiết cho {city}: {err}.\n"
                    "Vui lòng cấu hình biến môi trường `OPENWEATHER_API_KEY` trên server và khởi động lại `ai-assistant`."
                )
            else:
                aqi = get_aqi_data(city)
                aqi_val = int(aqi.get('aqi', 0)) if isinstance(aqi, dict) else 0
                category = get_aqi_category(aqi_val)
                recommendation = generate_recommendation(
                    aqi_val,
                    weather.get('temp', 0.0),
                    weather.get('description', ''),
                    weather.get('humidity', 0)
                )

                reply_text = (
                    f"Thời tiết {city} — {weather.get('city_name', city)}\n"
                    f"Nhiệt độ: {weather.get('temp')}°C (Feels like {weather.get('feels_like')}°C)\n"
                    f"Độ ẩm: {weather.get('humidity')}%\n"
                    f"Mô tả: {weather.get('description')}\n"
                    f"AQI: {aqi_val} ({category}), PM2.5: {aqi.get('pm25', 'n/a')}, PM10: {aqi.get('pm10', 'n/a')}\n"
                    f"Khuyến nghị: {recommendation}"
                )
            used_city = city
        else:
            # Fallback to generator for general queries
            prompt = text or 'Chào bạn, tôi có thể giúp gì?'
            ad_text = generate_ad_text(prompt)
            reply_text = ad_text

        # Save interaction to Drive (if configured)
        record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'user_id': str(chat_id) if chat_id else None,
            'incoming_text': text,
            'reply_text': reply_text,
            'intent_city': used_city
        }
        drive_file_id = None
        try:
            folder_id = os.getenv('GDRIVE_FOLDER_ID')
            if folder_id:
                drive_file_id = save_json_to_drive(record, folder_id)
        except Exception as e:
            print('Drive save failed:', e)

        return {"success": True, "reply": reply_text, "drive_file_id": drive_file_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get('/weather/neighbor-prob')
def neighbor_prob_proxy(station: str, csv_path: str = None, metric: str = 'pm25', min_overlap: int = 120, top_k: int = 3, precip_threshold: float = 0.5):
    """Proxy endpoint: forward to `app.weather.neighbor_probability` if available."""
    if neighbor_probability is None:
        raise HTTPException(status_code=404, detail='neighbor-prob not available')
    return neighbor_probability(station=station, csv_path=csv_path, metric=metric, min_overlap=min_overlap, top_k=top_k, precip_threshold=precip_threshold)
