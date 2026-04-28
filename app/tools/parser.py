import json
import uuid
from typing import Any, Dict, List, Optional

from ..utils import extract_json_like_content, safe_json_loads
from .schema import tool_call_names, tool_name_from_choice


def normalize_planned_tool_call(call: Any, allowed_names: List[str], selected_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(call, dict):
        return None

    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(call.get("name") or function.get("name") or "").strip()
    if not name:
        return None
    if allowed_names and name not in allowed_names:
        return None
    if selected_name and name != selected_name:
        return None

    arguments = call.get("arguments")
    if arguments is None:
        arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments_text = arguments
    else:
        arguments_text = json.dumps(arguments or {}, ensure_ascii=False)

    call_id = str(call.get("id") or call.get("call_id") or call.get("tool_call_id") or f"call_{uuid.uuid4().hex}")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments_text,
        },
    }


def normalize_tool_plan_output(content: str, tools: List[Dict[str, Any]], tool_choice: Any) -> Optional[Dict[str, Any]]:
    parsed = extract_json_like_content(content)
    if parsed is None:
        return None

    allowed_names = tool_call_names(tools)
    selected_name = tool_name_from_choice(tool_choice)
    require_tool = tool_choice == "required" or selected_name is not None

    if isinstance(parsed, list):
        parsed = {"type": "tool_call", "tool_calls": parsed}

    if not isinstance(parsed, dict):
        return None

    if "name" in parsed and "arguments" in parsed and "tool_calls" not in parsed and "tool_call" not in parsed:
        parsed = {"type": "tool_call", "tool_calls": [parsed]}

    tool_calls = parsed.get("tool_calls")
    if tool_calls is None and parsed.get("tool_call") is not None:
        tool_calls = [parsed.get("tool_call")]
    if tool_calls is None and parsed.get("type") == "tool_call":
        tool_calls = [parsed]

    if tool_calls:
        normalized_calls: List[Dict[str, Any]] = []
        for call in tool_calls:
            normalized = normalize_planned_tool_call(call, allowed_names, selected_name)
            if normalized is None:
                return None
            normalized_calls.append(normalized)
        if not normalized_calls:
            return None
        return {"kind": "tool_calls", "tool_calls": normalized_calls}

    final_content = parsed.get("content")
    if final_content is None:
        final_content = parsed.get("final")
    if final_content is None:
        final_content = parsed.get("answer")
    if final_content is None and isinstance(parsed.get("text"), str):
        final_content = parsed.get("text")

    if final_content is None:
        return None
    if require_tool:
        return None
    return {"kind": "final", "content": str(final_content)}


def apply_tool_plan_to_completion(completion: Dict[str, Any], tools: List[Dict[str, Any]], tool_choice: Any) -> Optional[Dict[str, Any]]:
    choice = (completion.get("choices") or [{}])[0]
    message = dict(choice.get("message") or {})
    content = str(message.get("content") or "")
    if not content.strip():
        return None

    plan = normalize_tool_plan_output(content, tools, tool_choice)
    if plan is None:
        return None

    if plan["kind"] == "tool_calls":
        message["content"] = None
        message["tool_calls"] = plan["tool_calls"]
        choice = {**choice, "message": message, "finish_reason": "tool_calls"}
    else:
        message["content"] = plan["content"]
        message.pop("tool_calls", None)
        choice = {**choice, "message": message, "finish_reason": "stop"}

    return {**completion, "choices": [choice, *list((completion.get("choices") or [])[1:])]}
