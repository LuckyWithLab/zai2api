from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..converters import (
    assistant_message_from_completion,
    responses_from_chat_completion,
    responses_input_to_messages,
    stream_completion_from_responses_response,
)
from ..models import ChatCompletionRequest, ResponsesRequest
from ..tools.schema import normalize_tool_definitions, request_uses_tools
from ..zai.client import collect_nonempty_completion, collect_structured_tool_completion
from .common import execute_with_auth, openai_error, response_store, store_response_transcript
from .models import resolve_model

router = APIRouter()


@router.post("/v1/responses")
async def responses(request: ResponsesRequest, req: Request):
    request.model = resolve_model(request.model)
    normalized_tools = normalize_tool_definitions(request.tools)
    tool_mode = request_uses_tools(normalized_tools, request.tool_choice)
    previous_messages = []
    if request.previous_response_id:
        response_store.cleanup()
        previous_messages = response_store.get(request.previous_response_id) or []
        if not previous_messages:
            return openai_error(f"Unknown previous_response_id: {request.previous_response_id}", 404)

    input_messages = responses_input_to_messages(request.input, request.instructions, previous_messages)
    include_usage = bool((request.stream_options or {}).get("include_usage"))
    chat_request = ChatCompletionRequest(
        model=request.model,
        messages=input_messages,
        stream=request.stream,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        tools=request.tools,
        tool_choice=request.tool_choice,
        stream_options=request.stream_options,
    )

    async def call_upstream(client, auth_token, chat_id):
        if tool_mode:
            completion = await collect_structured_tool_completion(
                client, auth_token, chat_id, chat_request, include_usage, normalized_tools
            )
            if completion is None:
                return openai_error("Model did not emit a valid structured tool plan.", 500)
        else:
            completion = await collect_nonempty_completion(
                client, auth_token, chat_id, chat_request, include_usage
            )

        response = responses_from_chat_completion(completion)
        store_response_transcript(str(response["id"]), input_messages, assistant_message_from_completion(completion))

        if request.stream:
            return StreamingResponse(
                stream_completion_from_responses_response(completion, include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return JSONResponse(content=response)

    try:
        result = await execute_with_auth(req, request.model, input_messages, call_upstream)
        return result
    except HTTPException as exc:
        return openai_error(str(exc.detail), exc.status_code)
    except Exception as exc:
        return openai_error(str(exc), 500)
