"""
API数据模型定义
符合OpenAI API规范的数据模型
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import time


class ToolCallFunction(BaseModel):
    """
    工具调用中函数信息

    - ``name``: 函数名称
    - ``arguments``: 传递给函数的JSON字符串参数
    """
    name: str = Field(..., description="函数名称")
    arguments: str = Field(..., description="函数参数(JSON字符串)")


class ToolCall(BaseModel):
    """
    单个工具调用对象

    OpenAI工具调用响应中会包含一个或多个工具调用，每个调用具有唯一id、类型和函数信息。
    """
    id: str = Field(..., description="调用ID")
    type: str = Field(..., description="调用类型，当前始终为function")
    function: ToolCallFunction = Field(..., description="被调用的函数信息")


class ToolCallDelta(BaseModel):
    """
    流式响应中的工具调用增量

    在流式返回中，工具调用信息可能会分块返回，各字段均为可选。
    """
    id: Optional[str] = Field(None, description="调用ID")
    type: Optional[str] = Field(None, description="调用类型")
    function: Optional[ToolCallFunction] = Field(None, description="函数信息")


class ChatMessage(BaseModel):
    """聊天消息模型

    ``role`` 字段可取 ``system``、``user``、``assistant`` 或 ``tool``。当模型调用工具时，
    会在返回消息中将 ``role`` 设置为 ``assistant``，并通过 ``tool_calls`` 字段返回结构化的调用信息，此时 ``content`` 可能为空或 ``None``。
    """
    role: str = Field(..., description="消息角色: system, user, assistant, tool")
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="消息内容，可以是字符串或内容块列表")
    name: Optional[str] = Field(None, description="消息发送者名称")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="工具调用列表，当模型调用工具时返回")
    tool_call_id: Optional[str] = Field(None, description="当role为tool时，标识对应的工具调用ID")
    function_call: Optional[Dict[str, Any]] = Field(None, description="旧版函数调用结构，用于兼容function_call字段")


class ChatCompletionRequest(BaseModel):
    """聊天补全请求模型"""
    model: str = Field(..., description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    
    # 可选参数
    stream: Optional[bool] = Field(False, description="是否流式响应")
    temperature: Optional[float] = Field(1.0, description="温度参数", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="最大token数", gt=0)
    top_p: Optional[float] = Field(1.0, description="top_p参数", ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(0.0, description="频率惩罚", ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, description="存在惩罚", ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = Field(None, description="停止序列")
    n: Optional[int] = Field(1, description="生成的回复数量", ge=1, le=10)
    user: Optional[str] = Field(None, description="用户标识")

    # 工具调用相关参数
    tools: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="可供模型调用的工具列表。每个工具定义必须包含 type='function' 和对应的 function 对象（包含 name、description、parameters 等）。"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description=(
            "控制模型调用工具行为的策略。可选值: "
            "'auto'（默认，模型决定是否以及何时调用工具），"
            "'none'（禁止调用工具），"
            "'required'（强制模型调用至少一个工具），"
            "或指定特定函数的对象，如{'type': 'function', 'function': {'name': 'get_weather'}}。"
        )
    )
    # 向后兼容旧版函数调用参数
    functions: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="旧版函数调用定义，与tools参数功能类似，将自动映射到tools。"
    )
    function_call: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description="旧版函数调用策略，将自动映射到tool_choice。"
    )


class Usage(BaseModel):
    """Token使用统计模型"""
    prompt_tokens: int = Field(..., description="输入token数")
    completion_tokens: int = Field(..., description="输出token数")  
    total_tokens: int = Field(..., description="总token数")


class ChatCompletionChoice(BaseModel):
    """聊天补全选择模型"""
    index: int = Field(..., description="选择索引")
    message: ChatMessage = Field(..., description="回复消息")
    finish_reason: Optional[str] = Field(None, description="完成原因")


class ChatCompletionResponse(BaseModel):
    """聊天补全响应模型"""
    id: str = Field(..., description="请求ID")
    object: str = Field("chat.completion", description="对象类型")
    created: int = Field(default_factory=lambda: int(time.time()), description="创建时间戳")
    model: str = Field(..., description="使用的模型")
    choices: List[ChatCompletionChoice] = Field(..., description="回复选择列表")
    usage: Optional[Usage] = Field(None, description="Token使用统计")


class DeltaMessage(BaseModel):
    """流式响应增量消息模型"""
    role: Optional[str] = Field(None, description="消息角色")
    content: Optional[str] = Field(None, description="增量内容")
    function_call: Optional[Dict[str, Any]] = Field(None, description="旧版函数调用增量信息")
    tool_calls: Optional[List[ToolCallDelta]] = Field(
        None,
        description="工具调用增量列表。在流式响应中，当模型决定调用工具时，该字段用于逐步返回调用信息。"
    )


class ChatCompletionChunkChoice(BaseModel):
    """流式响应块选择模型"""
    index: int = Field(..., description="选择索引")
    delta: DeltaMessage = Field(..., description="增量消息")
    finish_reason: Optional[str] = Field(None, description="完成原因")


class ChatCompletionChunk(BaseModel):
    """流式响应块模型"""
    id: str = Field(..., description="请求ID")
    object: str = Field("chat.completion.chunk", description="对象类型")
    created: int = Field(default_factory=lambda: int(time.time()), description="创建时间戳")
    model: str = Field(..., description="使用的模型")
    choices: List[ChatCompletionChunkChoice] = Field(..., description="增量选择列表")


class Model(BaseModel):
    """模型信息模型"""
    id: str = Field(..., description="模型ID")
    object: str = Field("model", description="对象类型")
    created: int = Field(default_factory=lambda: int(time.time()), description="创建时间戳")
    owned_by: str = Field("llmcallgateway", description="模型所有者")


class ModelList(BaseModel):
    """模型列表响应模型"""
    object: str = Field("list", description="对象类型")
    data: List[Model] = Field(..., description="模型列表")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: Dict[str, Any] = Field(..., description="错误信息")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    service: str = Field("LLMCallGateway", description="服务名称")
    status: str = Field("running", description="服务状态")
    version: str = Field("2.0.0", description="服务版本")
    description: str = Field("LLM API代理服务", description="服务描述")
    timestamp: int = Field(default_factory=lambda: int(time.time()), description="时间戳")


class MetricsResponse(BaseModel):
    """指标响应模型"""
    total_requests: int = Field(..., description="总请求数")
    total_tokens: int = Field(..., description="总token数")
    average_latency: float = Field(..., description="平均延迟(秒)")
    success_rate: float = Field(..., description="成功率")
    models_used: Dict[str, int] = Field(..., description="模型使用统计")


# === Embeddings相关模型 ===

class EmbeddingRequest(BaseModel):
    """
    Embeddings请求模型

    注意：input字段必须是原始文本字符串，不能是tokenized的数字数组！

    正确示例：
    - 单个文本: {"input": "Hello world", "model": "text-embedding-3-small"}
    - 多个文本: {"input": ["Hello", "World"], "model": "text-embedding-3-small"}

    错误示例：
    - ❌ 不要发送tokenized数组: {"input": [3134, 419, 57086], ...}
    - ❌ 不要发送嵌套数组: {"input": [[3134, 419]], ...}
    """
    input: Union[str, List[str]] = Field(
        ...,
        description="要嵌入的原始文本或文本列表。必须是字符串格式，不接受tokenized数字数组。",
        examples=[
            "Hello world",
            ["Hello", "World", "How are you?"]
        ]
    )
    model: str = Field(..., description="嵌入模型名称，如: text-embedding-3-small")
    encoding_format: Optional[str] = Field("float", description="编码格式: 'float'返回浮点数组，'base64'返回base64编码字符串")
    dimensions: Optional[int] = Field(None, description="嵌入向量维度", gt=0)
    user: Optional[str] = Field(None, description="用户标识")


class EmbeddingData(BaseModel):
    """
    单个嵌入数据模型

    embedding字段格式取决于请求中的encoding_format参数：
    - encoding_format='float' (默认): 返回List[float]浮点数组
    - encoding_format='base64': 返回str base64编码字符串
    """
    object: str = Field("embedding", description="对象类型")
    embedding: Union[List[float], str] = Field(..., description="嵌入向量(float数组或base64字符串)")
    index: int = Field(..., description="在输入列表中的索引")


class EmbeddingResponse(BaseModel):
    """Embeddings响应模型"""
    object: str = Field("list", description="对象类型")
    data: List[EmbeddingData] = Field(..., description="嵌入数据列表")
    model: str = Field(..., description="使用的模型")
    usage: Usage = Field(..., description="Token使用统计")


# 请求上下文模型
class RequestContext(BaseModel):
    """请求上下文"""
    request_id: str = Field(..., description="请求ID")
    start_time: float = Field(..., description="开始时间")
    model: str = Field(..., description="模型名称")
    user_id: Optional[str] = Field(None, description="用户ID")
    stream: bool = Field(False, description="是否流式")
    request_type: str = Field("chat", description="请求类型: chat, embedding")

    class Config:
        arbitrary_types_allowed = True
