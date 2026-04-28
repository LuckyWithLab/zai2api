import asyncio
import base64
import json
import time
import uuid
from typing import Optional

import httpx

from ..config import BASE_URL, DEFAULT_HEADERS

# Guest token 缓存
cached_token: Optional[str] = None
cached_token_expires_at: float = 0.0

# 认证 token 缓存
cached_auth_token: Optional[str] = None
cached_auth_token_expires_at: float = 0.0

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


def clear_cached_auth_token() -> None:
    global cached_auth_token, cached_auth_token_expires_at
    cached_auth_token = None
    cached_auth_token_expires_at = 0.0


async def _try_login() -> Optional[str]:
    """尝试通过浏览器登录获取认证 token"""
    try:
        from .login import login_and_get_token
        token = await login_and_get_token()
        if token:
            payload = _decode_token_payload(token) or {}
            exp = payload.get("exp")
            return token, float(exp) if isinstance(exp, (int, float)) else 0.0
        return None, 0.0
    except Exception as e:
        print(f"[!] 登录异常: {e}")
        return None, 0.0


async def get_guest_token(client: Optional[httpx.AsyncClient] = None, force_refresh: bool = False) -> Optional[str]:
    """获取 token，优先使用认证 token，其次 guest token"""
    global cached_token, cached_token_expires_at
    global cached_auth_token, cached_auth_token_expires_at

    async with _token_lock:
        # 优先用认证 token
        if _token_is_valid(cached_auth_token, cached_auth_token_expires_at):
            return cached_auth_token

        if force_refresh:
            clear_cached_token()
            clear_cached_auth_token()

        if _token_is_valid(cached_token, cached_token_expires_at):
            return cached_token

        url = f"{BASE_URL}/api/v1/auths/"
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
        try:
            # 先尝试 guest token
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

            # Guest token 失败，尝试登录
            print("[!] Guest token 失败，尝试登录...")
            auth_token, auth_exp = await _try_login()
            if auth_token:
                cached_auth_token = auth_token
                cached_auth_token_expires_at = auth_exp
                return cached_auth_token

            return None
        finally:
            if owns_client:
                await client.aclose()


def get_cached_token() -> Optional[str]:
    """获取缓存的 token（优先认证 token）"""
    if _token_is_valid(cached_auth_token, cached_auth_token_expires_at):
        return cached_auth_token
    if _token_is_valid(cached_token, cached_token_expires_at):
        return cached_token
    clear_cached_token()
    clear_cached_auth_token()
    return None


def extract_user_id(token: str) -> str:
    payload = _decode_token_payload(token) or {}
    return str(payload.get("id") or uuid.uuid4())
