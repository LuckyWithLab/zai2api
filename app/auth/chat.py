import asyncio
import time
import uuid
from typing import Optional

import httpx

from ..config import BASE_URL, DEFAULT_HEADERS


async def get_chat_id(client: httpx.AsyncClient, auth_token: str, model: str, message: str) -> str:
    url = f"{BASE_URL}/api/v1/chats/new"
    msg_id = str(uuid.uuid4())
    now_ts = int(time.time())
    payload = {
        "chat": {
            "id": "",
            "title": "新聊天",
            "models": [model],
            "params": {},
            "history": {
                "messages": {
                    msg_id: {
                        "id": msg_id,
                        "parentId": None,
                        "childrenIds": [],
                        "role": "user",
                        "content": message,
                        "timestamp": now_ts,
                        "models": [model],
                    }
                },
                "currentId": msg_id,
            },
            "tags": [],
            "flags": [],
            "features": [{"type": "tool_selector", "server": "tool_selector_h", "status": "hidden"}],
            "mcp_servers": ["advanced-search"],
            "enable_thinking": True,
            "auto_web_search": True,
            "message_version": 1,
            "extra": {},
            "timestamp": now_ts * 1000,
            "type": "default",
        }
    }
    headers = {**DEFAULT_HEADERS, "Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                return resp.json()["id"]
            last_error = Exception(f"Failed to create chat: {resp.text}")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            last_error = e
        if attempt < 2:
            await asyncio.sleep(2**attempt)
    raise last_error or Exception("Failed to create chat")
