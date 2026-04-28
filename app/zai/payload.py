import time
import uuid
from typing import Any, Dict, List, Tuple

from ..auth.signature import generate_signature
from ..auth.token import extract_user_id
from ..config import BASE_URL, DEFAULT_HEADERS
from ..models import ChatCompletionRequest
from ..tools.schema import normalize_tool_definitions
from ..tools.prompt import build_tool_prompt, prepend_tool_prompt
from ..utils import message_content, message_role, normalize_stop, safe_json_loads


def normalize_request_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
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


def build_zai_payload(request: ChatCompletionRequest, auth_token: str, chat_id: str) -> Tuple[Dict[str, Any], str, Dict[str, Any], bool]:
    messages, signature_prompt = normalize_request_messages(request.messages)
    tools = normalize_tool_definitions(request.tools)
    tool_choice = request.tool_choice
    if tools and tool_choice is None:
        tool_choice = "auto"

    if tools and tool_choice != "none":
        tool_prompt = build_tool_prompt(tools, request.tool_choice)
        messages = prepend_tool_prompt(messages, tool_prompt)

    # 默认 features 列表（隐藏的 MCP 工具）
    DEFAULT_COMPLETION_FEATURES = [
        {"type": "mcp", "server": "vibe-coding", "status": "hidden"},
        {"type": "mcp", "server": "ppt-maker", "status": "hidden"},
        {"type": "mcp", "server": "image-search", "status": "hidden"},
        {"type": "mcp", "server": "deep-research", "status": "hidden"},
        {"type": "tool_selector", "server": "tool_selector", "status": "hidden"},
        {"type": "mcp", "server": "advanced-search", "status": "hidden"},
    ]

    features = {
        "image_generation": False,
        "web_search": False,
        "auto_web_search": False,
        "preview_mode": True,
        "flags": [],
        "features": DEFAULT_COMPLETION_FEATURES,
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

    message_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    payload = {
        "stream": bool(request.stream),
        "model": request.model,
        "messages": messages,
        "signature_prompt": signature_prompt,
        "files": [],
        "params": params,
        "extra": {},
        "features": features,
        "background_tasks": {"title_generation": True, "tags_generation": True},
        "mcp_servers": [],
        "variables": {
            "{{USER_NAME}}": "Guest",
            "{{USER_LOCATION}}": "Unknown",
            "{{CURRENT_DATETIME}}": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "{{CURRENT_DATE}}": time.strftime("%Y-%m-%d", time.localtime()),
            "{{CURRENT_TIME}}": time.strftime("%H:%M:%S", time.localtime()),
            "{{CURRENT_WEEKDAY}}": time.strftime("%A", time.localtime()),
            "{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
            "{{USER_LANGUAGE}}": "zh-CN",
        },
        "model_item": {
            "id": request.model,
            "name": request.model,
            "owned_by": "zai",
        },
        "chat_id": chat_id,
        "id": message_id,
        "session_id": session_id,
        "current_user_message_id": message_id,
        "current_user_message_parent_id": None,
    }

    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    else:
        payload["tools"] = None

    return payload, signature_prompt, {}, bool(tools and tool_choice != "none")


def zai_stream_request_headers(auth_token: str, signature: str) -> Dict[str, str]:
    return {
        **DEFAULT_HEADERS,
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "Accept": "*/*",
        "X-Signature": signature,
    }


def zai_request_query(auth_token: str, chat_id: str, now_ms: int, req_id: str, user_id: str) -> Dict[str, str]:
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


def prepare_request(request: ChatCompletionRequest, auth_token: str, chat_id: str):
    now_ms = int(time.time() * 1000)
    user_id = extract_user_id(auth_token)
    req_id = str(uuid.uuid4())
    payload, signature_prompt, _, _ = build_zai_payload(request, auth_token, chat_id)
    signature = generate_signature(signature_prompt, str(now_ms), req_id, user_id)
    headers = zai_stream_request_headers(auth_token, signature)
    query = zai_request_query(auth_token, chat_id, now_ms, req_id, user_id)
    return payload, headers, query
