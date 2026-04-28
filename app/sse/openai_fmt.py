from typing import Any, Dict, Optional

from ..models import StreamState


def openai_chunk(
    state: StreamState,
    delta: Optional[Dict[str, Any]] = None,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    usage_chunk: bool = False,
) -> Dict[str, Any]:
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
