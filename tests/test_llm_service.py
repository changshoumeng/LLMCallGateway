import json

from app.models.api_models import (
    ChatCompletionRequest,
    ChatMessage,
    ToolCall,
    ToolCallFunction,
)
from app.services.llm_service import LLMService


def _build_service() -> LLMService:
    """Helper to instantiate service without relying on global state."""
    return LLMService()


def test_prepare_litellm_request_includes_tool_calls_and_results():
    service = _build_service()

    tool_definition = {
        "type": "function",
        "function": {
            "name": "search_songs",
            "description": "搜索歌曲",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }

    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        tools=[tool_definition],
        messages=[
            ChatMessage(role="system", content="你是音乐助理"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=ToolCallFunction(
                            name="search_songs",
                            arguments=json.dumps({"query": "Beyond"}),
                        ),
                    )
                ],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="call_1",
                content=json.dumps({"songs": ["光辉岁月"]}, ensure_ascii=False),
            ),
        ],
    )

    payload = service._prepare_litellm_request(request)
    assert payload["model"] == "gpt-4o-mini"
    assistant_msg = payload["messages"][1]
    tool_msg = payload["messages"][2]

    assert assistant_msg["tool_calls"][0]["function"]["name"] == "search_songs"
    assert assistant_msg["content"] is None
    assert tool_msg["tool_call_id"] == "call_1"
    assert payload["tools"][0]["function"]["name"] == "search_songs"


def test_prepare_litellm_request_accepts_legacy_function_call_messages():
    service = _build_service()

    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[
            ChatMessage(role="system", content="legacy"),
            ChatMessage(
                role="assistant",
                content=None,
                function_call={
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Shanghai"}),
                },
            ),
        ],
    )

    payload = service._prepare_litellm_request(request)
    assistant_msg = payload["messages"][1]

    assert assistant_msg["function_call"]["name"] == "get_weather"
    assert assistant_msg["content"] is None


def test_extract_tool_calls_returns_models_and_logs():
    service = _build_service()

    class Message:
        def __init__(self):
            self.tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_songs",
                        "arguments": '{"query": "Beyond"}',
                    },
                }
            ]

    models, logs = service._extract_tool_calls(Message())

    assert logs is not None and logs[0]["function"]["name"] == "search_songs"
    assert models is not None and models[0].function.arguments == '{"query": "Beyond"}'


def test_extract_function_call_payload_keeps_arguments_string():
    service = _build_service()

    class Message:
        def __init__(self):
            self.function_call = {"name": "get_weather", "arguments": {"city": "Shanghai"}}

    payload = service._extract_function_call_payload(Message())

    assert payload == {"name": "get_weather", "arguments": "{'city': 'Shanghai'}"}
