import json
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import clear_cached_token, get_cached_token, get_chat_id, get_guest_token
from .config import API_KEY
from .models import ChatCompletionRequest, ResponsesRequest, StreamState
from .stream import yield_openai_sse, openai_chunk
from .tools import normalize_tool_definitions, request_uses_tools
from .upstream import (
    collect_nonempty_completion,
    collect_structured_tool_completion,
    _normalize_request_messages,
)
from .utils import clone_messages, flatten_content, message_role

app = FastAPI(title="GLM-5.1 OpenAI Proxy")

# 全局 httpx 客户端（连接复用）
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=120.0)
    return _http_client


@app.on_event("shutdown")
async def shutdown():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()


# response_store 带 TTL 淘汰（默认 30 分钟）
class TTLStore:
    def __init__(self, max_size: int = 1000, ttl: float = 1800):
        self._store: OrderedDict[str, tuple] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: List[Dict[str, Any]]):
        self._store[key] = (value, time.time())
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def cleanup(self):
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]


response_store = TTLStore(max_size=1000, ttl=1800)


# API Key 鉴权中间件
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if API_KEY:
        # /docs 和 /openapi.json 放行
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


def _openai_error(message: str, status_code: int = 500) -> JSONResponse:
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


def _store_response_transcript(response_id: str, messages: List[Dict[str, Any]], assistant_message: Dict[str, Any]) -> None:
    response_store.cleanup()
    response_store.set(response_id, [*clone_messages(messages), dict(assistant_message)])


def _extract_token_from_header(header_value: Optional[str]) -> str:
    if not header_value:
        return ""
    return header_value[7:] if header_value.startswith("Bearer ") else header_value.strip()


def _resolve_upstream_token(request: Request) -> tuple[Optional[str], bool]:
    for header_name in ("X-Upstream-Authorization", "X-ZAI-Authorization"):
        token = _extract_token_from_header(request.headers.get(header_name))
        if token:
            return token, True
    if not API_KEY:
        token = _extract_token_from_header(request.headers.get("Authorization"))
        if token:
            return token, True
    return get_cached_token(), False


def _assistant_message_from_completion(completion: Dict[str, Any]) -> Dict[str, Any]:
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    assistant_message: Dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content"),
    }
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        assistant_message["tool_calls"] = [dict(tool_call) for tool_call in tool_calls if isinstance(tool_call, dict)]
    return assistant_message


def _responses_input_item_to_messages(item: Any) -> List[Dict[str, Any]]:
    if item is None:
        return []
    if isinstance(item, str):
        return [{"role": "user", "content": item}]
    if not isinstance(item, dict):
        return [{"role": "user", "content": flatten_content(item)}]

    item_type = item.get("type")
    role = message_role(item)

    if item_type in {"function_call_output", "tool_result"} or role == "tool":
        tool_call_id = str(item.get("call_id") or item.get("tool_call_id") or item.get("id") or "")
        content = item.get("output") or item.get("content") or item.get("text")
        return [
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": item.get("name"),
                "content": flatten_content(content),
                "metadata": item.get("metadata"),
            }
        ]

    if item_type == "function_call":
        arguments = item.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        tool_call_id = str(item.get("call_id") or item.get("id") or "")
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": str(arguments or ""),
                        },
                    }
                ],
            }
        ]

    if item_type == "message" or role in {"user", "assistant", "system"}:
        content = item.get("content") or item.get("text")
        return [{"role": role or item.get("role") or "user", "content": flatten_content(content)}]

    content = item.get("content") or item.get("text")
    return [{"role": "user", "content": flatten_content(content)}]


def _responses_input_to_messages(input_data: Any, instructions: Optional[str] = None, previous_messages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if previous_messages:
        messages.extend(clone_messages(previous_messages))
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})
    if isinstance(input_data, list):
        for item in input_data:
            messages.extend(_responses_input_item_to_messages(item))
    else:
        messages.extend(_responses_input_item_to_messages(input_data))
    return messages


def _responses_from_chat_completion(completion: Dict[str, Any]) -> Dict[str, Any]:
    response_id = str(completion.get("id") or f"resp_{uuid.uuid4().hex}")
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        output = [
            {
                "id": str(tc.get("id") or uuid.uuid4()),
                "type": "function_call",
                "call_id": str(tc.get("id") or uuid.uuid4()),
                "name": (tc.get("function") or {}).get("name"),
                "arguments": (tc.get("function") or {}).get("arguments") or "",
                "status": "completed",
            }
            for tc in tool_calls
        ]
        status = "requires_action"
    else:
        content = message.get("content") or ""
        output = [
            {
                "id": str(uuid.uuid4()),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content}],
            }
        ]
        status = "completed"

    response = {
        "id": response_id,
        "object": "response",
        "created_at": completion.get("created"),
        "model": completion.get("model"),
        "status": status,
        "output": output,
    }
    if completion.get("usage") is not None:
        response["usage"] = completion["usage"]
    return response


def _stream_completion_from_chat_response(completion: Dict[str, Any], include_usage: bool):
    state = StreamState(
        response_id=str(completion.get("id") or f"chatcmpl-{uuid.uuid4()}"),
        created=int(completion.get("created") or time.time()),
        model=str(completion.get("model") or ""),
    )
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    tool_calls = message.get("tool_calls") or []
    finish_reason = str(choice.get("finish_reason") or ("tool_calls" if tool_calls else "stop"))

    yield yield_openai_sse(openai_chunk(state, delta={"role": "assistant"}))

    if tool_calls:
        tool_deltas = [
            {
                "index": i,
                "id": str(tc.get("id") or uuid.uuid4()),
                "type": "function",
                "function": {k: v for k in ("name", "arguments") if (v := (tc.get("function") or {}).get(k)) is not None},
            }
            for i, tc in enumerate(tool_calls)
        ]
        yield yield_openai_sse(openai_chunk(state, delta={"tool_calls": tool_deltas}))
    elif content:
        yield yield_openai_sse(openai_chunk(state, delta={"content": content}))

    if include_usage and completion.get("usage") is not None:
        yield yield_openai_sse(openai_chunk(state, usage=completion["usage"], usage_chunk=True))
    yield yield_openai_sse(openai_chunk(state, finish_reason=finish_reason))
    yield "data: [DONE]\n\n"


async def _stream_completion_from_responses_response(completion: Dict[str, Any], include_usage: bool):
    response = _responses_from_chat_completion(completion)
    created_at = int(response.get("created_at") or time.time())

    yield yield_openai_sse({
        "type": "response.created",
        "response": {"id": response["id"], "object": "response", "created_at": created_at, "model": response.get("model"), "status": "in_progress"},
    })

    output = response.get("output") or []
    if response.get("status") == "requires_action":
        for item in output:
            call_id = str(item.get("call_id") or item.get("id") or uuid.uuid4())
            yield yield_openai_sse({
                "type": "response.output_item.added",
                "response_id": response["id"],
                "item": {"id": call_id, "type": "function_call", "call_id": call_id, "name": item.get("name"), "arguments": str(item.get("arguments") or ""), "status": "in_progress"},
            })
            yield yield_openai_sse({
                "type": "response.function_call_arguments.delta",
                "response_id": response["id"], "item_id": call_id, "delta": str(item.get("arguments") or ""),
            })
    else:
        text = ""
        if output:
            content = (output[0].get("content") or [])
            if content:
                text = str(content[0].get("text") or "")
        if text:
            yield yield_openai_sse({"type": "response.output_text.delta", "response_id": response["id"], "delta": text})

    final_response = dict(response)
    if include_usage and completion.get("usage") is not None:
        final_response["usage"] = completion["usage"]
    yield yield_openai_sse({"type": "response.completed", "response": final_response})
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    include_usage = bool((request.stream_options or {}).get("include_usage"))
    normalized_tools = normalize_tool_definitions(request.tools)
    tool_mode = request_uses_tools(normalized_tools, request.tool_choice)

    try:
        client = await get_http_client()
        auth_token, explicit_upstream_token = _resolve_upstream_token(req)
        if not auth_token:
            auth_token = await get_guest_token(client=client)
        if not auth_token:
            return _openai_error("Unable to obtain a guest token from Z.ai", 502)

        chat_hint = _normalize_request_messages(request.messages)[1]
        try:
            chat_id = await get_chat_id(client, auth_token, request.model, chat_hint)
        except Exception:
            if explicit_upstream_token:
                raise
            clear_cached_token()
            auth_token = await get_guest_token(client=client, force_refresh=True)
            if not auth_token:
                return _openai_error("Unable to obtain a guest token from Z.ai", 502)
            chat_id = await get_chat_id(client, auth_token, request.model, chat_hint)

        try:
            if tool_mode:
                completion = await collect_structured_tool_completion(client, auth_token, chat_id, request, include_usage, normalized_tools)
                if completion is None:
                    return _openai_error("Model did not emit a valid structured tool plan.", 500)
            else:
                completion = await collect_nonempty_completion(client, auth_token, chat_id, request, include_usage)
        except HTTPException as exc:
            if explicit_upstream_token or exc.status_code not in {401, 403}:
                raise
            clear_cached_token()
            auth_token = await get_guest_token(client=client, force_refresh=True)
            if not auth_token:
                return _openai_error("Unable to obtain a guest token from Z.ai", 502)
            chat_id = await get_chat_id(client, auth_token, request.model, chat_hint)
            if tool_mode:
                completion = await collect_structured_tool_completion(client, auth_token, chat_id, request, include_usage, normalized_tools)
                if completion is None:
                    return _openai_error("Model did not emit a valid structured tool plan.", 500)
            else:
                completion = await collect_nonempty_completion(client, auth_token, chat_id, request, include_usage)

        if request.stream:
            return StreamingResponse(
                _stream_completion_from_chat_response(completion, include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        return JSONResponse(content=completion)
    except HTTPException as exc:
        return _openai_error(str(exc.detail), exc.status_code)
    except Exception as exc:
        return _openai_error(str(exc), 500)


@app.post("/v1/responses")
async def responses(request: ResponsesRequest, req: Request):
    normalized_tools = normalize_tool_definitions(request.tools)
    tool_mode = request_uses_tools(normalized_tools, request.tool_choice)
    previous_messages = []
    if request.previous_response_id:
        response_store.cleanup()
        previous_messages = response_store.get(request.previous_response_id) or []
        if not previous_messages:
            return _openai_error(f"Unknown previous_response_id: {request.previous_response_id}", 404)

    input_messages = _responses_input_to_messages(request.input, request.instructions, previous_messages)
    include_usage = bool((request.stream_options or {}).get("include_usage"))
    chat_request = ChatCompletionRequest(
        model=request.model,
        messages=input_messages,
        stream=request.stream,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        tools=request.tools,
        tool_choice=request.tool_choice,
        stream_options=request.stream_options,
    )

    try:
        client = await get_http_client()
        auth_token, explicit_upstream_token = _resolve_upstream_token(req)
        if not auth_token:
            auth_token = await get_guest_token(client=client)
        if not auth_token:
            return _openai_error("Unable to obtain a guest token from Z.ai", 502)

        chat_hint = _normalize_request_messages(input_messages)[1]
        try:
            chat_id = await get_chat_id(client, auth_token, request.model, chat_hint)
        except Exception:
            if explicit_upstream_token:
                raise
            clear_cached_token()
            auth_token = await get_guest_token(client=client, force_refresh=True)
            if not auth_token:
                return _openai_error("Unable to obtain a guest token from Z.ai", 502)
            chat_id = await get_chat_id(client, auth_token, request.model, chat_hint)

        try:
            if tool_mode:
                completion = await collect_structured_tool_completion(client, auth_token, chat_id, chat_request, include_usage, normalized_tools)
                if completion is None:
                    return _openai_error("Model did not emit a valid structured tool plan.", 500)
            else:
                completion = await collect_nonempty_completion(client, auth_token, chat_id, chat_request, include_usage)
        except HTTPException as exc:
            if explicit_upstream_token or exc.status_code not in {401, 403}:
                raise
            clear_cached_token()
            auth_token = await get_guest_token(client=client, force_refresh=True)
            if not auth_token:
                return _openai_error("Unable to obtain a guest token from Z.ai", 502)
            chat_id = await get_chat_id(client, auth_token, request.model, chat_hint)
            if tool_mode:
                completion = await collect_structured_tool_completion(client, auth_token, chat_id, chat_request, include_usage, normalized_tools)
                if completion is None:
                    return _openai_error("Model did not emit a valid structured tool plan.", 500)
            else:
                completion = await collect_nonempty_completion(client, auth_token, chat_id, chat_request, include_usage)

        response = _responses_from_chat_completion(completion)
        _store_response_transcript(str(response["id"]), input_messages, _assistant_message_from_completion(completion))

        if request.stream and not tool_mode:
            return StreamingResponse(
                _stream_completion_from_responses_response(completion, include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        if request.stream:
            return StreamingResponse(
                _stream_completion_from_responses_response(completion, include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return JSONResponse(content=response)
    except HTTPException as exc:
        return _openai_error(str(exc.detail), exc.status_code)
    except Exception as exc:
        return _openai_error(str(exc), 500)
