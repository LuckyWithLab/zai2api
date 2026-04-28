import json
import time
import uuid
from typing import Any, Dict, List, Optional

from .models import ChatCompletionRequest, ResponsesRequest, StreamState
from .sse.parser import yield_openai_sse
from .sse.openai_fmt import openai_chunk
from .utils import clone_messages, flatten_content, message_role


def assistant_message_from_completion(completion: Dict[str, Any]) -> Dict[str, Any]:
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    assistant_message: Dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content"),
    }
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        assistant_message["tool_calls"] = [
            dict(tool_call) for tool_call in tool_calls if isinstance(tool_call, dict)]
    return assistant_message


def responses_input_item_to_messages(item: Any) -> List[Dict[str, Any]]:
    if item is None:
        return []
    if isinstance(item, str):
        return [{"role": "user", "content": item}]
    if not isinstance(item, dict):
        return [{"role": "user", "content": flatten_content(item)}]

    item_type = item.get("type")
    role = message_role(item)

    if item_type in {"function_call_output", "tool_result"} or role == "tool":
        tool_call_id = str(item.get("call_id") or item.get(
            "tool_call_id") or item.get("id") or "")
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


def responses_input_to_messages(input_data: Any, instructions: Optional[str] = None, previous_messages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if previous_messages:
        messages.extend(clone_messages(previous_messages))
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})
    if isinstance(input_data, list):
        for item in input_data:
            messages.extend(responses_input_item_to_messages(item))
    else:
        messages.extend(responses_input_item_to_messages(input_data))
    return messages


def responses_from_chat_completion(completion: Dict[str, Any]) -> Dict[str, Any]:
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


def stream_completion_from_chat_response(completion: Dict[str, Any], include_usage: bool):
    state = StreamState(
        response_id=str(completion.get("id") or f"chatcmpl-{uuid.uuid4()}"),
        created=int(completion.get("created") or time.time()),
        model=str(completion.get("model") or ""),
    )
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    tool_calls = message.get("tool_calls") or []
    finish_reason = str(choice.get("finish_reason") or (
        "tool_calls" if tool_calls else "stop"))

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


async def stream_completion_from_responses_response(completion: Dict[str, Any], include_usage: bool):
    response = responses_from_chat_completion(completion)
    created_at = int(response.get("created_at") or time.time())

    yield yield_openai_sse({
        "type": "response.created",
        "response": {"id": response["id"], "object": "response", "created_at": created_at, "model": response.get("model"), "status": "in_progress"},
    })

    output = response.get("output") or []
    if response.get("status") == "requires_action":
        for item in output:
            call_id = str(item.get("call_id")
                          or item.get("id") or uuid.uuid4())
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
