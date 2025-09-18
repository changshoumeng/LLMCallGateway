"""
API数据模型定义
符合OpenAI API规范的数据模型
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import time


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色: system, user, assistant")
    content: str = Field(..., description="消息内容")
    name: Optional[str] = Field(None, description="消息发送者名称")


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


# 请求上下文模型
class RequestContext(BaseModel):
    """请求上下文"""
    request_id: str = Field(..., description="请求ID")
    start_time: float = Field(..., description="开始时间")
    model: str = Field(..., description="模型名称")
    user_id: Optional[str] = Field(None, description="用户ID")
    stream: bool = Field(False, description="是否流式")
    
    class Config:
        arbitrary_types_allowed = True