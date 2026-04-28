import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_URL = os.getenv("ZAI_BASE_URL", "https://chat.z.ai")
SECRET = os.getenv("ZAI_SECRET", "")
API_KEY = os.getenv("API_KEY", "")

DEFAULT_HEADERS = {
    "Host": "chat.z.ai",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0",
    "Accept": "application/json",
    "Accept-Language": "zh-CN",
    "X-FE-Version": "prod-fe-1.1.12",
    "Origin": "https://chat.z.ai",
    "Referer": "https://chat.z.ai/",
    "Connection": "keep-alive",
}
