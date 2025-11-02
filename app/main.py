import os
import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException

from app.schemas import GenerateRequest, GenerateResponse
from app.gdrive import save_json_to_drive
from app.zalo import send_message_to_zalo
from app.weather import router as weather_router

try:
    import openai
except Exception:
    openai = None

APP_PORT = int(os.getenv("APP_PORT", "8000"))

app = FastAPI(title="AI Zalo Assistant + Weather Forecast")

# Include weather router
app.include_router(weather_router)


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
