import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from ..config import BASE_URL
from ..models import ChatCompletionRequest, StreamState
from ..sse.parser import extract_zai_data, yield_openai_sse
from ..sse.openai_fmt import openai_chunk, openai_final_response
from ..sse.state import apply_zai_event, maybe_finish_tool_turn
from ..tools.parser import apply_tool_plan_to_completion
from ..tools.prompt import build_tool_repair_prompt, prepend_repair_prompt
from .errors import (
    upstream_cooldown_seconds,
    upstream_sse_error,
    should_retry_upstream_error,
    upstream_cooldown_until,
)
from .payload import prepare_request


async def stream_zai_completion(
    client: httpx.AsyncClient,
    auth_token: str,
    chat_id: str,
    request: ChatCompletionRequest,
    include_usage: bool,
):
    from .errors import upstream_cooldown_until as _unused  # noqa: F401 — keep module loaded

    payload, headers, query = prepare_request(request, auth_token, chat_id)
    response_state = StreamState(response_id=f"chatcmpl-{uuid.uuid4()}", created=int(time.time()), model=request.model)

    async with client.stream(
        "POST",
        f"{BASE_URL}/api/v2/chat/completions",
        params=query,
        json=payload,
        headers=headers,
    ) as response:
        if response.status_code != 200:
            detail = (await response.aread()).decode()
            raise HTTPException(status_code=response.status_code, detail=detail or "Upstream request failed")

        async for line in response.aiter_lines():
            z_data = extract_zai_data(line or "")
            if not z_data:
                continue
            error_info = upstream_sse_error(z_data)
            if error_info is not None:
                status_code, detail = error_info
                raise HTTPException(status_code=status_code, detail=detail)
            emitted, should_stop = apply_zai_event(response_state, z_data, include_usage)
            for item in emitted:
                if item.get("done"):
                    yield "data: [DONE]\n\n"
                    return
                yield yield_openai_sse(item)
            if should_stop:
                yield "data: [DONE]\n\n"
                return

    if response_state.seen_tool_call:
        finished, finish_chunks = maybe_finish_tool_turn(response_state, include_usage)
        if finished:
            for item in finish_chunks:
                if item.get("done"):
                    continue
                yield yield_openai_sse(item)
    elif response_state.text() or response_state.usage is not None:
        if include_usage and response_state.usage is not None:
            yield yield_openai_sse(openai_chunk(response_state, usage=response_state.usage, usage_chunk=True))
        yield yield_openai_sse(openai_chunk(response_state, finish_reason="stop"))

    yield "data: [DONE]\n\n"


async def collect_zai_completion(
    client: httpx.AsyncClient,
    auth_token: str,
    chat_id: str,
    request: ChatCompletionRequest,
    include_usage: bool,
) -> Dict[str, Any]:
    import app.zai.errors as errors_module

    max_attempts = 3
    for attempt in range(max_attempts):
        now = time.time()
        if errors_module.upstream_cooldown_until > now:
            await asyncio.sleep(errors_module.upstream_cooldown_until - now)

        payload, headers, query = prepare_request(request, auth_token, chat_id)
        response_state = StreamState(response_id=f"chatcmpl-{uuid.uuid4()}", created=int(time.time()), model=request.model)

        try:
            async with client.stream(
                "POST",
                f"{BASE_URL}/api/v2/chat/completions",
                params=query,
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode()
                    cooldown = upstream_cooldown_seconds(response.status_code, detail or "", attempt)
                    if cooldown > 0:
                        errors_module.upstream_cooldown_until = max(errors_module.upstream_cooldown_until, time.time() + cooldown)
                    if attempt + 1 < max_attempts and should_retry_upstream_error(response.status_code, detail or ""):
                        await asyncio.sleep(2**attempt)
                        continue
                    raise HTTPException(status_code=response.status_code, detail=detail or "Upstream request failed")

                async for line in response.aiter_lines():
                    z_data = extract_zai_data(line or "")
                    if not z_data:
                        continue
                    error_info = upstream_sse_error(z_data)
                    if error_info is not None:
                        status_code, detail = error_info
                        cooldown = upstream_cooldown_seconds(status_code, detail, attempt)
                        if cooldown > 0:
                            errors_module.upstream_cooldown_until = max(errors_module.upstream_cooldown_until, time.time() + cooldown)
                        raise HTTPException(status_code=status_code, detail=detail)
                    _, should_stop = apply_zai_event(response_state, z_data, include_usage)
                    if should_stop:
                        break
        except httpx.RequestError as exc:
            if attempt + 1 < max_attempts:
                await asyncio.sleep(2**attempt)
                continue
            raise HTTPException(status_code=503, detail=str(exc) or "Upstream request failed")

        finish_reason = "tool_calls" if response_state.seen_tool_call else "stop"
        return openai_final_response(response_state, finish_reason, include_usage)

    raise HTTPException(status_code=503, detail="Upstream request repeatedly failed")


async def collect_nonempty_completion(
    client: httpx.AsyncClient,
    auth_token: str,
    chat_id: str,
    request: ChatCompletionRequest,
    include_usage: bool,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    completion: Dict[str, Any] = {}
    for attempt in range(max_attempts):
        completion = await collect_zai_completion(client, auth_token, chat_id, request.model_copy(update={"stream": False}), include_usage)
        choice = (completion.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        if content.strip():
            return completion
        if (message.get("tool_calls") or []) and content.strip() == "":
            return completion
        if attempt + 1 < max_attempts:
            request = request.model_copy(
                update={
                    "stream": False,
                    "messages": prepend_repair_prompt(
                        request.messages,
                        "Your previous reply was empty. Reply with a non-empty answer in plain text.",
                    ),
                }
            )
    return completion


async def collect_structured_tool_completion(
    client: httpx.AsyncClient,
    auth_token: str,
    chat_id: str,
    request: ChatCompletionRequest,
    include_usage: bool,
    normalized_tools: List[Dict[str, Any]],
    max_attempts: int = 2,
) -> Optional[Dict[str, Any]]:
    for attempt in range(max_attempts):
        attempt_request = request.model_copy(update={"stream": False})
        if attempt > 0:
            repair_prompt = build_tool_repair_prompt(normalized_tools, request.tool_choice)
            attempt_request = request.model_copy(
                update={
                    "stream": False,
                    "messages": prepend_repair_prompt(request.messages, repair_prompt),
                }
            )
        try:
            completion = await collect_zai_completion(client, auth_token, chat_id, attempt_request, include_usage)
            if completion is not None:
                parsed = apply_tool_plan_to_completion(completion, normalized_tools, request.tool_choice)
                if parsed is not None:
                    return parsed
        except HTTPException as exc:
            if attempt + 1 < max_attempts and exc.status_code in {405, 429, 500, 502, 503, 504}:
                await asyncio.sleep(2**attempt)
                continue
            raise
        except httpx.RequestError:
            if attempt + 1 < max_attempts:
                await asyncio.sleep(2**attempt)
                continue
            raise
    return None
