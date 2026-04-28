import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import HTTPException

from .auth import extract_user_id, generate_signature
from .config import BASE_URL, DEFAULT_HEADERS
from .models import ChatCompletionRequest, StreamState
from .stream import apply_zai_event, extract_zai_data, openai_chunk, openai_final_response, yield_openai_sse
from .tools import (
    apply_tool_plan_to_completion,
    build_tool_prompt,
    build_tool_repair_prompt,
    normalize_tool_definitions,
    prepend_repair_prompt,
    prepend_tool_prompt,
    request_uses_tools,
)
from .utils import message_content, message_role, normalize_stop, safe_json_loads

upstream_cooldown_until: float = 0.0


def _upstream_sse_error(z_data: Dict[str, Any]) -> Optional[Tuple[int, str]]:
    error = z_data.get("error") if isinstance(z_data.get("error"), dict) else None
    if not error:
        return None
    code = str(error.get("code") or "").upper()
    detail = str(error.get("detail") or error.get("message") or code or "Upstream request failed")
    if code == "MODEL_CONCURRENCY_LIMIT":
        return 429, detail
    return 502, detail


def _should_retry_upstream_error(status_code: int, detail: str) -> bool:
    if status_code in {429, 500, 502, 503, 504}:
        return True
    if status_code != 405:
        return False
    normalized = detail.lower()
    return any(
        marker in normalized
        for marker in (
            "blocked as it may cause potential threats",
            "model_concurrency_limit",
            "internal server error",
            "oops, something went wrong",
        )
    )


def _upstream_cooldown_seconds(status_code: int, detail: str, attempt: int) -> float:
    normalized = detail.lower()
    if status_code == 405:
        return min(30.0 * (attempt + 1), 90.0)
    if "model_concurrency_limit" in normalized:
        return min(10.0 * (attempt + 1), 30.0)
    if status_code in {429, 500, 502, 503, 504}:
        return min(3.0 * (attempt + 1), 15.0)
    return 0.0


def _normalize_request_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    converted: List[Dict[str, Any]] = []
    last_non_empty_prompt = ""
    last_user_prompt = ""

    for message in messages:
        role = message_role(message)
        content = message_content(message)
        if content.strip():
            last_non_empty_prompt = content.strip()
        if role == "user" and content.strip():
            last_user_prompt = content.strip()

        if role == "assistant":
            converted_message: Dict[str, Any] = {"role": "assistant", "content": content if content else ""}
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                converted_tool_calls: Dict[str, Dict[str, Any]] = {}
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    call_id = str(call.get("id") or uuid.uuid4())
                    function = call.get("function") or {}
                    if not isinstance(function, dict):
                        function = {}
                    converted_tool_calls[call_id] = {
                        "id": call_id,
                        "name": str(function.get("name") or ""),
                        "arguments": str(function.get("arguments") or ""),
                        "parsed_arguments": safe_json_loads(str(function.get("arguments") or "")),
                    }
                if converted_tool_calls:
                    converted_message["tool_calls"] = converted_tool_calls
            converted.append(converted_message)
            continue

        if role == "tool":
            converted.append({
                "role": "tool",
                "tool_call_id": str(message.get("tool_call_id") or ""),
                "name": message.get("name"),
                "content": content,
                "metadata": message.get("metadata"),
            })
            continue

        converted.append({"role": role or "user", "content": content})

    signature_prompt = last_user_prompt or last_non_empty_prompt or "你好"
    return converted, signature_prompt


def _build_zai_payload(request: ChatCompletionRequest, auth_token: str, chat_id: str) -> Tuple[Dict[str, Any], str, Dict[str, Any], bool]:
    messages, signature_prompt = _normalize_request_messages(request.messages)
    tools = normalize_tool_definitions(request.tools)
    tool_choice = request.tool_choice
    if tools and tool_choice is None:
        tool_choice = "auto"

    if tools and tool_choice != "none":
        tool_prompt = build_tool_prompt(tools, request.tool_choice)
        messages = prepend_tool_prompt(messages, tool_prompt)

    features = {
        "image_generation": False,
        "web_search": False,
        "auto_web_search": False,
        "preview_mode": True,
        "flags": [],
        "vlm_tools_enable": False,
        "vlm_web_search_enable": False,
        "vlm_website_mode": False,
        "enable_thinking": True,
    }

    params: Dict[str, Any] = {}
    if request.temperature is not None:
        params["temperature"] = request.temperature
    if request.top_p is not None:
        params["top_p"] = request.top_p
    if request.max_tokens is not None:
        params["max_tokens"] = request.max_tokens
    stop = normalize_stop(request.stop)
    if stop:
        params["stop"] = stop

    payload = {
        "stream": bool(request.stream),
        "tool_stream": False,
        "model": request.model,
        "messages": messages,
        "signature_prompt": signature_prompt,
        "params": params,
        "extra": {},
        "tools": None,
        "tool_choice": None,
        "mcp_servers": None,
        "features": features,
        "variables": {
            "{{USER_NAME}}": f"Guest-{int(time.time() * 1000)}",
            "{{USER_LOCATION}}": "Unknown",
            "{{CURRENT_DATETIME}}": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "{{CURRENT_DATE}}": time.strftime("%Y-%m-%d", time.localtime()),
            "{{CURRENT_TIME}}": time.strftime("%H:%M:%S", time.localtime()),
            "{{CURRENT_WEEKDAY}}": time.strftime("%A", time.localtime()),
            "{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
            "{{USER_LANGUAGE}}": "zh-CN",
        },
        "chat_id": chat_id,
        "id": str(uuid.uuid4()),
        "current_user_message_id": str(uuid.uuid4()),
        "current_user_message_parent_id": None,
        "background_tasks": {"title_generation": True, "tags_generation": True},
    }

    return payload, signature_prompt, {}, bool(tools and tool_choice != "none")


def _zai_stream_request_headers(auth_token: str, signature: str) -> Dict[str, str]:
    return {
        **DEFAULT_HEADERS,
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Signature": signature,
    }


def _zai_request_query(auth_token: str, chat_id: str, now_ms: int, req_id: str, user_id: str) -> Dict[str, str]:
    return {
        "timestamp": str(now_ms),
        "requestId": req_id,
        "user_id": user_id,
        "version": "0.0.1",
        "platform": "web",
        "token": auth_token,
        "user_agent": DEFAULT_HEADERS["User-Agent"],
        "language": "zh-CN",
        "languages": "zh-CN,zh,zh-TW,zh-HK,en-US,en",
        "timezone": "Asia/Shanghai",
        "cookie_enabled": "true",
        "screen_width": "1920",
        "screen_height": "1080",
        "screen_resolution": "1920x1080",
        "viewport_height": "1080",
        "viewport_width": "1920",
        "viewport_size": "1920x1080",
        "color_depth": "24",
        "pixel_ratio": "1",
        "current_url": f"https://chat.z.ai/c/{chat_id}",
        "pathname": f"/c/{chat_id}",
        "search": "",
        "hash": "",
        "host": "chat.z.ai",
        "hostname": "chat.z.ai",
        "protocol": "https:",
        "referrer": "",
        "title": "Z.ai - Free AI Chatbot & Agent powered by GLM-5.1 & GLM-5",
        "timezone_offset": "-480",
        "local_time": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "utc_time": time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()),
        "is_mobile": "false",
        "is_touch": "false",
        "max_touch_points": "0",
        "browser_name": "Firefox",
        "os_name": "Linux",
        "signature_timestamp": str(now_ms),
    }


def _prepare_request(request: ChatCompletionRequest, auth_token: str, chat_id: str):
    now_ms = int(time.time() * 1000)
    user_id = extract_user_id(auth_token)
    req_id = str(uuid.uuid4())
    payload, signature_prompt, _, _ = _build_zai_payload(request, auth_token, chat_id)
    signature = generate_signature(signature_prompt, str(now_ms), req_id, user_id)
    headers = _zai_stream_request_headers(auth_token, signature)
    query = _zai_request_query(auth_token, chat_id, now_ms, req_id, user_id)
    return payload, headers, query


async def stream_zai_completion(client: httpx.AsyncClient, auth_token: str, chat_id: str, request: ChatCompletionRequest, include_usage: bool):
    payload, headers, query = _prepare_request(request, auth_token, chat_id)
    response_state = StreamState(response_id=f"chatcmpl-{uuid.uuid4()}", created=int(time.time()), model=request.model)

    async with client.stream(
        "POST",
        f"{BASE_URL}/api/v2/chat/completions",
        params=query,
        json=payload,
        headers=headers,
    ) as response:
        if response.status_code != 200:
            detail = (await response.aread()).decode()
            raise HTTPException(status_code=response.status_code, detail=detail or "Upstream request failed")

        async for line in response.aiter_lines():
            z_data = extract_zai_data(line or "")
            if not z_data:
                continue
            error_info = _upstream_sse_error(z_data)
            if error_info is not None:
                status_code, detail = error_info
                raise HTTPException(status_code=status_code, detail=detail)
            emitted, should_stop = apply_zai_event(response_state, z_data, include_usage)
            for item in emitted:
                if item.get("done"):
                    yield "data: [DONE]\n\n"
                    return
                yield yield_openai_sse(item)
            if should_stop:
                yield "data: [DONE]\n\n"
                return

    if response_state.seen_tool_call:
        from .stream import maybe_finish_tool_turn
        finished, finish_chunks = maybe_finish_tool_turn(response_state, include_usage)
        if finished:
            for item in finish_chunks:
                if item.get("done"):
                    continue
                yield yield_openai_sse(item)
    elif response_state.text() or response_state.usage is not None:
        if include_usage and response_state.usage is not None:
            yield yield_openai_sse(openai_chunk(response_state, usage=response_state.usage, usage_chunk=True))
        yield yield_openai_sse(openai_chunk(response_state, finish_reason="stop"))

    yield "data: [DONE]\n\n"


async def collect_zai_completion(client: httpx.AsyncClient, auth_token: str, chat_id: str, request: ChatCompletionRequest, include_usage: bool) -> Dict[str, Any]:
    global upstream_cooldown_until
    max_attempts = 3
    for attempt in range(max_attempts):
        now = time.time()
        if upstream_cooldown_until > now:
            await asyncio.sleep(upstream_cooldown_until - now)

        payload, headers, query = _prepare_request(request, auth_token, chat_id)
        response_state = StreamState(response_id=f"chatcmpl-{uuid.uuid4()}", created=int(time.time()), model=request.model)

        try:
            async with client.stream(
                "POST",
                f"{BASE_URL}/api/v2/chat/completions",
                params=query,
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode()
                    cooldown = _upstream_cooldown_seconds(response.status_code, detail or "", attempt)
                    if cooldown > 0:
                        upstream_cooldown_until = max(upstream_cooldown_until, time.time() + cooldown)
                    if attempt + 1 < max_attempts and _should_retry_upstream_error(response.status_code, detail or ""):
                        await asyncio.sleep(2**attempt)
                        continue
                    raise HTTPException(status_code=response.status_code, detail=detail or "Upstream request failed")

                async for line in response.aiter_lines():
                    z_data = extract_zai_data(line or "")
                    if not z_data:
                        continue
                    error_info = _upstream_sse_error(z_data)
                    if error_info is not None:
                        status_code, detail = error_info
                        cooldown = _upstream_cooldown_seconds(status_code, detail, attempt)
                        if cooldown > 0:
                            upstream_cooldown_until = max(upstream_cooldown_until, time.time() + cooldown)
                        raise HTTPException(status_code=status_code, detail=detail)
                    _, should_stop = apply_zai_event(response_state, z_data, include_usage)
                    if should_stop:
                        break
        except httpx.RequestError as exc:
            if attempt + 1 < max_attempts:
                await asyncio.sleep(2**attempt)
                continue
            raise HTTPException(status_code=503, detail=str(exc) or "Upstream request failed")

        finish_reason = "tool_calls" if response_state.seen_tool_call else "stop"
        return openai_final_response(response_state, finish_reason, include_usage)

    raise HTTPException(status_code=503, detail="Upstream request repeatedly failed")


async def collect_nonempty_completion(client: httpx.AsyncClient, auth_token: str, chat_id: str, request: ChatCompletionRequest, include_usage: bool, max_attempts: int = 3) -> Dict[str, Any]:
    for attempt in range(max_attempts):
        completion = await collect_zai_completion(client, auth_token, chat_id, request.model_copy(update={"stream": False}), include_usage)
        choice = (completion.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        if content.strip():
            return completion
        if (message.get("tool_calls") or []) and content.strip() == "":
            return completion
        if attempt + 1 < max_attempts:
            request = request.model_copy(
                update={
                    "stream": False,
                    "messages": prepend_repair_prompt(
                        request.messages,
                        "Your previous reply was empty. Reply with a non-empty answer in plain text.",
                    ),
                }
            )
    return completion


async def collect_structured_tool_completion(
    client: httpx.AsyncClient,
    auth_token: str,
    chat_id: str,
    request: ChatCompletionRequest,
    include_usage: bool,
    normalized_tools: List[Dict[str, Any]],
    max_attempts: int = 2,
) -> Optional[Dict[str, Any]]:
    for attempt in range(max_attempts):
        attempt_request = request.model_copy(update={"stream": False})
        if attempt > 0:
            repair_prompt = build_tool_repair_prompt(normalized_tools, request.tool_choice)
            attempt_request = request.model_copy(
                update={
                    "stream": False,
                    "messages": prepend_repair_prompt(request.messages, repair_prompt),
                }
            )
        try:
            completion = await collect_zai_completion(client, auth_token, chat_id, attempt_request, include_usage)
            if completion is not None:
                parsed = apply_tool_plan_to_completion(completion, normalized_tools, request.tool_choice)
                if parsed is not None:
                    return parsed
        except HTTPException as exc:
            if attempt + 1 < max_attempts and exc.status_code in {405, 429, 500, 502, 503, 504}:
                await asyncio.sleep(2**attempt)
                continue
            raise
        except httpx.RequestError:
            if attempt + 1 < max_attempts:
                await asyncio.sleep(2**attempt)
                continue
            raise
    return None
