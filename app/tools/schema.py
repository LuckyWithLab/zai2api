from typing import Any, Dict, List, Optional


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
