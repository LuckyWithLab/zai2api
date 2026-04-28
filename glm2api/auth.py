import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Optional

import httpx

from .config import BASE_URL, DEFAULT_HEADERS, SECRET

cached_token: Optional[str] = None
cached_token_expires_at: float = 0.0
_token_lock = asyncio.Lock()


def _decode_token_payload(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload_part = parts[1]
        payload_part += "=" * (-len(payload_part) % 4)
        payload = base64.urlsafe_b64decode(payload_part.encode("ascii"))
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return None


def _token_is_valid(token: Optional[str], expires_at: float, skew_seconds: int = 60) -> bool:
    if not token:
        return False
    if expires_at <= 0:
        return True
    return time.time() + skew_seconds < expires_at


def clear_cached_token() -> None:
    global cached_token, cached_token_expires_at
    cached_token = None
    cached_token_expires_at = 0.0


async def get_guest_token(client: Optional[httpx.AsyncClient] = None, force_refresh: bool = False) -> Optional[str]:
    global cached_token, cached_token_expires_at
    async with _token_lock:
        if force_refresh:
            clear_cached_token()
        if _token_is_valid(cached_token, cached_token_expires_at):
            return cached_token
        url = f"{BASE_URL}/api/v1/auths/"
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
        try:
            for attempt in range(3):
                try:
                    resp = await client.get(url, headers=DEFAULT_HEADERS)
                    if resp.status_code == 200:
                        data = resp.json()
                        token = data.get("token")
                        if token:
                            payload = _decode_token_payload(token) or {}
                            exp = payload.get("exp")
                            cached_token = token
                            cached_token_expires_at = float(exp) if isinstance(exp, (int, float)) else 0.0
                            return cached_token
                except (httpx.RequestError, httpx.TimeoutException) as e:
                    print(f"[!] 获取 Token 失败: {e}")
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
            return None
        finally:
            if owns_client:
                await client.aclose()


def get_cached_token() -> Optional[str]:
    if _token_is_valid(cached_token, cached_token_expires_at):
        return cached_token
    clear_cached_token()
    return None


def extract_user_id(token: str) -> str:
    payload = _decode_token_payload(token) or {}
    return str(payload.get("id") or uuid.uuid4())


def generate_signature(prompt: str, timestamp: str, request_id: str, user_id: str) -> str:
    sign_fields = {
        "requestId": request_id,
        "timestamp": timestamp,
        "user_id": user_id,
    }
    sorted_payload = ",".join(
        part
        for kv in sorted(sign_fields.items(), key=lambda x: x[0])
        for part in kv
    )

    prompt_b64 = base64.b64encode(prompt.strip().encode("utf-8")).decode("ascii")
    bucket = str(int(timestamp) // (5 * 60 * 1000))

    round1 = hmac.new(
        SECRET.encode("utf-8"),
        bucket.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    msg = f"{sorted_payload}|{prompt_b64}|{timestamp}"
    signature = hmac.new(
        round1.encode("utf-8"),
        msg.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return signature


async def get_chat_id(client: httpx.AsyncClient, auth_token: str, model: str, message: str) -> str:
    url = f"{BASE_URL}/api/v1/chats/new"
    msg_id = str(uuid.uuid4())
    now_ts = int(time.time())
    payload = {
        "chat": {
            "id": "",
            "title": "新聊天",
            "models": [model],
            "params": {},
            "history": {
                "messages": {
                    msg_id: {
                        "id": msg_id,
                        "parentId": None,
                        "childrenIds": [],
                        "role": "user",
                        "content": message,
                        "timestamp": now_ts,
                        "models": [model],
                    }
                },
                "currentId": msg_id,
            },
            "tags": [],
            "flags": [],
            "features": [{"type": "tool_selector", "server": "tool_selector_h", "status": "hidden"}],
            "mcp_servers": ["advanced-search"],
            "enable_thinking": True,
            "auto_web_search": True,
            "message_version": 1,
            "extra": {},
            "timestamp": now_ts * 1000,
            "type": "default",
        }
    }
    headers = {**DEFAULT_HEADERS, "Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                return resp.json()["id"]
            last_error = Exception(f"Failed to create chat: {resp.text}")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            last_error = e
        if attempt < 2:
            await asyncio.sleep(2**attempt)
    raise last_error or Exception("Failed to create chat")
