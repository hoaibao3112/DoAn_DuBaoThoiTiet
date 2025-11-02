from typing import Optional
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None


class GenerateResponse(BaseModel):
    success: bool
    ad_text: str
    drive_file_id: Optional[str] = None
