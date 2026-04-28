from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, List, Dict, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from ..auth.token import clear_cached_token, get_cached_token, get_guest_token
from ..auth.chat import get_chat_id
from ..cache import TTLStore
from ..config import API_KEY
from ..utils import clone_messages
from ..zai.payload import normalize_request_messages


# ── HTTP 客户端生命周期 ──

_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=120.0)
    return _http_client


async def _shutdown():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await _shutdown()


# ── Response Store (TTL) ──

response_store = TTLStore(max_size=1000, ttl=1800)


def store_response_transcript(response_id: str, messages: List[Dict[str, Any]], assistant_message: Dict[str, Any]) -> None:
    response_store.cleanup()
    response_store.set(response_id, [*clone_messages(messages), dict(assistant_message)])


# ── Error Helpers ──

def openai_error(message: str, status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "server_error" if status_code >= 500 else "invalid_request_error",
                "code": status_code,
            }
        },
    )


# ── Token Resolution ──

def _extract_token_from_header(header_value: Optional[str]) -> str:
    if not header_value:
        return ""
    return header_value[7:] if header_value.startswith("Bearer ") else header_value.strip()


def resolve_upstream_token(request: Request) -> tuple[Optional[str], bool]:
    for header_name in ("X-Upstream-Authorization", "X-ZAI-Authorization"):
        token = _extract_token_from_header(request.headers.get(header_name))
        if token:
            return token, True
    if not API_KEY:
        token = _extract_token_from_header(request.headers.get("Authorization"))
        if token:
            return token, True
    return get_cached_token(), False


# ── Auth Middleware ──

async def auth_middleware(request: Request, call_next):
    if API_KEY:
        if request.url.path in ("/docs", "/openapi.json", "/redoc"):
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else ""
        if token != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid API key", "type": "authentication_error"}},
            )
    return await call_next(request)


# ── Unified Auth + Retry Execution ──

async def execute_with_auth(
    req: Request,
    model: str,
    messages: list,
    call_upstream: Callable[[httpx.AsyncClient, str, str], Awaitable[Any]],
) -> Any:
    client = await get_http_client()
    auth_token, explicit = resolve_upstream_token(req)
    if not auth_token:
        auth_token = await get_guest_token(client=client)
    if not auth_token:
        return openai_error("Unable to obtain a guest token from Z.ai", 502)

    chat_hint = normalize_request_messages(messages)[1]

    try:
        chat_id = await get_chat_id(client, auth_token, model, chat_hint)
    except Exception:
        if explicit:
            raise
        clear_cached_token()
        auth_token = await get_guest_token(client=client, force_refresh=True)
        if not auth_token:
            return openai_error("Unable to obtain a guest token from Z.ai", 502)
        chat_id = await get_chat_id(client, auth_token, model, chat_hint)

    try:
        return await call_upstream(client, auth_token, chat_id)
    except HTTPException as exc:
        if explicit or exc.status_code not in {401, 403}:
            raise
        clear_cached_token()
        auth_token = await get_guest_token(client=client, force_refresh=True)
        if not auth_token:
            return openai_error("Unable to obtain a guest token from Z.ai", 502)
        chat_id = await get_chat_id(client, auth_token, model, chat_hint)
        return await call_upstream(client, auth_token, chat_id)
