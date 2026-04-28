import json
import uuid
from typing import Any, Dict, Optional


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
    if not isinstance(metadata, dict):
        return str(uuid.uuid4())
    return str(metadata.get("tool_call_id") or metadata.get("sub_tool_call_id") or metadata.get("id") or uuid.uuid4())


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
