import json
import uuid
from typing import Any, Dict, List, Optional

from .utils import extract_json_like_content, safe_json_loads


def tool_name_from_choice(tool_choice: Any) -> Optional[str]:
    if not isinstance(tool_choice, dict):
        return None
    function = tool_choice.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        return str(name) if name else None
    return None


def normalize_tool_definitions(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not name:
            continue
        normalized.append(
            {
                "type": tool.get("type", "function"),
                "function": {
                    "name": str(name),
                    "description": function.get("description"),
                    "parameters": function.get("parameters") or {},
                },
            }
        )
    return normalized


def tool_call_names(tools: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for tool in tools:
        name = str(tool.get("function", {}).get("name") or "").strip()
        if name:
            names.append(name)
    return names


def request_uses_tools(tools: List[Dict[str, Any]], tool_choice: Any) -> bool:
    return bool(tools) and tool_choice != "none"


def build_tool_prompt(tools: List[Dict[str, Any]], tool_choice: Any) -> str:
    """构建工具调用提示词（中文，带 JSON 示例）"""
    if not tools:
        return ""

    selected_name = tool_name_from_choice(tool_choice)
    require_tool = tool_choice == "required" or selected_name is not None
    tool_names = []
    lines = [
        "你可以使用以下工具来完成任务。当需要调用工具时，只输出一个JSON对象，不要输出任何其他内容。",
        "",
        "输出格式（严格遵守）：",
        '{"type":"tool_call","tool_calls":[{"name":"工具名","arguments":{...}}]}',
        "",
        "如果不需要调用工具，直接回答：",
        '{"type":"final","content":"你的回答"}',
        "",
        "可用工具：",
    ]
    for tool in tools:
        fn = tool["function"]
        name = fn["name"]
        tool_names.append(name)
        lines.append(f"- {name}: {fn.get('description') or '无描述'}")
        lines.append(f"  参数: {json.dumps(fn.get('parameters') or {}, ensure_ascii=False)}")
    lines.append("")
    if selected_name:
        lines.append(f"请使用 {selected_name} 工具。")
    if require_tool:
        lines.append("你必须调用工具来完成任务。")
    lines.append("")
    if tool_names:
        first_tool = tool_names[0]
        lines.append(f'示例输出: {{"type":"tool_call","tool_calls":[{{"name":"{first_tool}","arguments":{{}}}}]}}')
    return "\n".join(lines)


def build_tool_repair_prompt(tools: List[Dict[str, Any]], tool_choice: Any) -> str:
    base_prompt = build_tool_prompt(tools, tool_choice)
    if not base_prompt:
        return ""
    lines = [
        "你上一次的回复无效。",
        "请重新回复，只输出一个有效的JSON对象，不要输出任何其他内容。",
        "不要重复上一次的回复。",
        base_prompt,
    ]
    return "\n".join(lines)


def prepend_tool_prompt(messages: List[Dict[str, Any]], tool_prompt: str) -> List[Dict[str, Any]]:
    """把 tool prompt 追加到最后一条 user message（GLM-5.1 不遵循 system prompt）"""
    if not tool_prompt:
        return messages

    from .utils import message_role, message_content

    messages = [dict(m) for m in messages]
    for idx in range(len(messages) - 1, -1, -1):
        if message_role(messages[idx]) == "user":
            original = message_content(messages[idx])
            messages[idx] = {
                **messages[idx],
                "content": f"{original}\n\n{tool_prompt}",
            }
            return messages

    return [{"role": "user", "content": tool_prompt}, *messages]


def prepend_repair_prompt(messages: List[Dict[str, Any]], repair_prompt: str) -> List[Dict[str, Any]]:
    if not repair_prompt:
        return messages
    return [{"role": "system", "content": repair_prompt}, *messages]


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
