import json
from typing import Any, Dict, List

from .schema import tool_name_from_choice


def build_tool_prompt(tools: List[Dict[str, Any]], tool_choice: Any) -> str:
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
    if not tool_prompt:
        return messages

    from ..utils import message_role, message_content

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
