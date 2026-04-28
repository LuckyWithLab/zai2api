from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    model: str = "GLM-5.1"
    messages: List[Dict[str, Any]]
    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Any] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    stream_options: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class ResponsesRequest(BaseModel):
    model: str = "GLM-5.1"
    input: Any
    instructions: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    previous_response_id: Optional[str] = None
    stream_options: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


@dataclass
class ToolCallState:
    tool_call_id: str
    name: str
    arguments: str = ""
    parsed_arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamState:
    response_id: str
    created: int
    model: str
    content_parts: List[str] = field(default_factory=list)
    tool_calls: List[ToolCallState] = field(default_factory=list)
    tool_call_index: Dict[str, int] = field(default_factory=dict)
    seen_tool_call: bool = False
    usage: Optional[Dict[str, Any]] = None
    first_delta_sent: bool = False

    def add_text(self, text: str) -> None:
        if text:
            self.content_parts.append(text)

    def text(self) -> str:
        return "".join(self.content_parts)

    def upsert_tool_call(self, tool_call_id: str, name: str, arguments: str, parsed_arguments: Dict[str, Any]) -> int:
        if tool_call_id in self.tool_call_index:
            idx = self.tool_call_index[tool_call_id]
            state = self.tool_calls[idx]
            if name:
                state.name = name
            if arguments:
                state.arguments += arguments
            if parsed_arguments:
                state.parsed_arguments = parsed_arguments
            return idx

        idx = len(self.tool_calls)
        self.tool_call_index[tool_call_id] = idx
        self.tool_calls.append(
            ToolCallState(
                tool_call_id=tool_call_id,
                name=name,
                arguments=arguments,
                parsed_arguments=parsed_arguments,
            )
        )
        return idx
