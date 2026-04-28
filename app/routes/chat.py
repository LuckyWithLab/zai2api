from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..converters import stream_completion_from_chat_response
from ..models import ChatCompletionRequest
from ..tools.schema import normalize_tool_definitions, request_uses_tools
from ..zai.client import collect_nonempty_completion, collect_structured_tool_completion
from .common import execute_with_auth, openai_error
from .models import resolve_model

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    request.model = resolve_model(request.model)
    include_usage = bool((request.stream_options or {}).get("include_usage"))
    normalized_tools = normalize_tool_definitions(request.tools)
    tool_mode = request_uses_tools(normalized_tools, request.tool_choice)

    async def call_upstream(client, auth_token, chat_id):
        if tool_mode:
            completion = await collect_structured_tool_completion(
                client, auth_token, chat_id, request, include_usage, normalized_tools
            )
            if completion is None:
                return openai_error("Model did not emit a valid structured tool plan.", 500)
        else:
            completion = await collect_nonempty_completion(
                client, auth_token, chat_id, request, include_usage
            )

        if request.stream:
            return StreamingResponse(
                stream_completion_from_chat_response(completion, include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return JSONResponse(content=completion)

    try:
        result = await execute_with_auth(req, request.model, request.messages, call_upstream)
        return result
    except HTTPException as exc:
        return openai_error(str(exc.detail), exc.status_code)
    except Exception as exc:
        return openai_error(str(exc), 500)
