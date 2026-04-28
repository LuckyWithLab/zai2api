import json
from typing import Any, Dict, List, Optional, Tuple

from .models import StreamState, ToolCallState
from .utils import safe_json_loads


def yield_openai_sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def extract_zai_data(raw_line: str) -> Optional[Dict[str, Any]]:
    if not raw_line.startswith("data: "):
        return None
    body = raw_line[6:].strip()
    if not body or body == "[DONE]":
        return None
    try:
        parsed = json.loads(body)
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed.get("data") if isinstance(parsed.get("data"), dict) else parsed
    return None


def tool_call_id_from_metadata(metadata: Optional[Dict[str, Any]]) -> str:
    import uuid
    if not isinstance(metadata, dict):
        return str(uuid.uuid4())
    return str(metadata.get("tool_call_id") or metadata.get("sub_tool_call_id") or metadata.get("id") or uuid.uuid4())


def openai_chunk(state: StreamState, delta: Optional[Dict[str, Any]] = None, finish_reason: Optional[str] = None, usage: Optional[Dict[str, Any]] = None, usage_chunk: bool = False) -> Dict[str, Any]:
    if usage_chunk:
        chunk = {
            "id": state.response_id,
            "object": "chat.completion.chunk",
            "created": state.created,
            "model": state.model,
            "choices": [],
        }
        if usage is not None:
            chunk["usage"] = usage
        return chunk

    chunk: Dict[str, Any] = {
        "id": state.response_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def openai_final_response(state: StreamState, finish_reason: str, include_usage: bool) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant"}
    content = state.text()
    if content:
        message["content"] = content
    elif state.tool_calls:
        message["content"] = None
    else:
        message["content"] = content

    if state.tool_calls:
        message["tool_calls"] = [
            {
                "id": item.tool_call_id,
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": item.arguments,
                },
            }
            for item in state.tool_calls
        ]

    response: Dict[str, Any] = {
        "id": state.response_id,
        "object": "chat.completion",
        "created": state.created,
        "model": state.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    if include_usage and state.usage is not None:
        response["usage"] = state.usage
    return response


def maybe_finish_tool_turn(state: StreamState, include_usage: bool) -> Tuple[bool, List[Dict[str, Any]]]:
    if not state.seen_tool_call:
        return False, []
    chunks = []
    if include_usage and state.usage is not None:
        chunks.append(openai_chunk(state, usage=state.usage, usage_chunk=True))
    chunks.append(openai_chunk(state, finish_reason="tool_calls"))
    chunks.append({"done": True})
    return True, chunks


def apply_zai_event(state: StreamState, z_data: Dict[str, Any], include_usage: bool) -> Tuple[List[Dict[str, Any]], bool]:
    phase = str(z_data.get("phase") or "other")
    delta_content = str(z_data.get("delta_content") or "")
    delta_name = str(z_data.get("delta_name") or "")
    delta_arguments = str(z_data.get("delta_arguments") or "")
    metadata = z_data.get("metadata") if isinstance(z_data.get("metadata"), dict) else {}
    usage = z_data.get("usage") if isinstance(z_data.get("usage"), dict) else None

    emitted: List[Dict[str, Any]] = []

    if usage:
        state.usage = usage

    if phase in {"answer", "thinking"}:
        if delta_content:
            if not state.first_delta_sent:
                emitted.append(openai_chunk(state, delta={"role": "assistant"}))
                state.first_delta_sent = True
            emitted.append(openai_chunk(state, delta={"content": delta_content}))
            if phase == "answer":
                state.add_text(delta_content)
        return emitted, False

    if phase == "tool_call":
        state.seen_tool_call = True
        tc_id = tool_call_id_from_metadata(metadata)
        idx = state.upsert_tool_call(tc_id, delta_name, delta_arguments, safe_json_loads(delta_arguments))
        if not state.first_delta_sent:
            emitted.append(openai_chunk(state, delta={"role": "assistant"}))
            state.first_delta_sent = True
        tool_call_delta: Dict[str, Any] = {
            "index": idx,
            "id": tc_id,
            "type": "function",
            "function": {},
        }
        if delta_name:
            tool_call_delta["function"]["name"] = delta_name
        if delta_arguments:
            tool_call_delta["function"]["arguments"] = delta_arguments
        emitted.append(openai_chunk(state, delta={"tool_calls": [tool_call_delta]}))
        return emitted, False

    if state.seen_tool_call:
        finished, finish_chunks = maybe_finish_tool_turn(state, include_usage)
        return finish_chunks, finished

    if phase == "other":
        return emitted, False

    if phase == "done":
        if include_usage and state.usage is not None:
            emitted.append(openai_chunk(state, usage=state.usage, usage_chunk=True))
        emitted.append(openai_chunk(state, finish_reason="stop"))
        return emitted, True

    return emitted, False


def extract_chat_chunks_from_sse_line(line: str) -> Optional[Dict[str, Any]]:
    if not line.startswith("data: "):
        return None
    body = line[6:].strip()
    if not body or body == "[DONE]":
        return None
    try:
        parsed = json.loads(body)
    except Exception:
        return None
    if parsed.get("object") == "chat.completion.chunk":
        return parsed
    return None
