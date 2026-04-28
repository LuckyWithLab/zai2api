from typing import Any, Dict, List, Tuple

from ..models import StreamState
from ..utils import safe_json_loads
from .openai_fmt import openai_chunk
from .parser import tool_call_id_from_metadata


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
