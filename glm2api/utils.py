import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


def safe_json_loads(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        return {}


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is not None:
                    parts.append(str(text))
                continue
            if item_type == "image_url":
                image_url = item.get("image_url") or {}
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if url:
                    parts.append(f"[image:{url}]")
                continue
            parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def message_role(message: Dict[str, Any]) -> str:
    return str(message.get("role") or "").strip()


def message_content(message: Dict[str, Any]) -> str:
    return flatten_content(message.get("content"))


def normalize_stop(stop: Any) -> Optional[List[str]]:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, (list, tuple)):
        items = [str(item) for item in stop if item is not None and str(item) != ""]
        return items or None
    return [str(stop)]


def clone_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(message) for message in messages]


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_like_content(text: str) -> Optional[Any]:
    if not text:
        return None

    candidates = []
    stripped = strip_code_fences(text)
    if stripped:
        candidates.append(stripped)
    match = re.search(r"<tool_call[^>]*>\s*(.*?)\s*</tool_call[^>]*>", text, re.S | re.I)
    if match:
        candidates.append(match.group(1).strip())
    if stripped != text.strip():
        candidates.append(text.strip())

    for candidate in candidates:
        decoder = json.JSONDecoder()
        starts = [0]
        brace_index = candidate.find("{")
        if brace_index >= 0:
            starts.append(brace_index)
        if candidate.lstrip().startswith("["):
            bracket_index = candidate.find("[")
            if bracket_index >= 0:
                starts.append(bracket_index)
        for start in sorted(set(starts)):
            snippet = candidate[start:].lstrip()
            try:
                parsed, _ = decoder.raw_decode(snippet)
                return parsed
            except Exception:
                continue
    return None


def generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex}"
