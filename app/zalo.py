import os
import requests


def send_message_to_zalo(user_id: str, text: str) -> dict:
    """Send a text message to a Zalo user via OA/chatbot API.

    Note: Ensure ZALO_ACCESS_TOKEN is set and the OA has permission to message the user.
    The default API URL can be overridden via ZALO_API_URL env var.
    """
    token = os.getenv("ZALO_ACCESS_TOKEN")
    if not token:
        raise EnvironmentError("ZALO_ACCESS_TOKEN not set")

    url = os.getenv("ZALO_API_URL", "https://openapi.zalo.me/v2.0/oa/message")

    payload = {
        "recipient": {"user_id": user_id},
        "message": {"type": "text", "text": text},
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, params={"access_token": token}, json=payload, headers=headers, timeout=10)
    try:
        return resp.json()
    except Exception:
        return {"status_code": resp.status_code, "text": resp.text}
