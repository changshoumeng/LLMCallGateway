"""
LLM代理服务
基于LiteLLM实现多模型统一接口，支持详细的交互日志和指标统计
"""

import uuid
import time
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
import json

import litellm
from litellm import completion, acompletion

from ..core.config import settings
from ..core.logging import system_logger, llm_interaction_logger
from ..models.api_models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk,
    ChatMessage, ChatCompletionChoice, ChatCompletionChunkChoice,
    DeltaMessage, Usage, RequestContext
)
from ..services.metrics import metrics_collector


class LLMService:
    """LLM代理服务"""
    
    def __init__(self):
        self.setup_litellm()
    
    def setup_litellm(self) -> None:
        """配置LiteLLM"""
        # 基础配置
        litellm.set_verbose = settings.debug
        litellm.drop_params = True  # 自动过滤不支持的参数
        litellm.request_timeout = settings.request_timeout
        
        # 如果有配置API密钥，设置默认密钥
        if settings.litellm_api_key:
            litellm.api_key = settings.litellm_api_key
        
        # 如果有配置基础URL，设置默认URL
        if settings.litellm_base_url:
            litellm.api_base = settings.litellm_base_url
        
        system_logger.info(f"LiteLLM配置完成 - Timeout: {settings.request_timeout}s")
    
    async def create_chat_completion(self, request: ChatCompletionRequest,
                                   user_id: Optional[str] = None) -> Union[ChatCompletionResponse, AsyncGenerator]:
        """创建聊天补全"""
        # 生成请求ID
        request_id = str(uuid.uuid4())[:8]
        
        # 创建请求上下文
        context = RequestContext(
            request_id=request_id,
            start_time=time.time(),
            model=request.model,
            user_id=user_id,
            stream=request.stream or False
        )
        
        # 开始记录指标
        metrics = metrics_collector.start_request(context)
        
        try:
            # 准备LiteLLM请求参数
            llm_request = self._prepare_litellm_request(request)
            
            # 开始记录下游交互
            llm_interaction_logger.start_interaction(
                request_id, "litellm", llm_request
            )
            
            if request.stream:
                # 流式响应
                return self._handle_stream_completion(
                    request_id, llm_request, context, metrics
                )
            else:
                # 非流式响应
                return await self._handle_non_stream_completion(
                    request_id, llm_request, context, metrics
                )
        
        except Exception as e:
            # 记录错误交互
            llm_interaction_logger.log_error_interaction(request_id, e, "create_chat_completion")
            
            # 完成指标记录
            metrics_collector.complete_request(
                request_id, success=False, error_message=str(e)
            )
            
            system_logger.error(f"聊天补全失败: {e}")
            raise e
    
    def _extract_user_query(self, messages: List[ChatMessage]) -> str:
        """提取用户查询内容"""
        user_messages = [msg.content for msg in messages if msg.role == "user"]
        if user_messages:
            return user_messages[-1]  # 返回最后一条用户消息
        return "无用户查询"
    
    def _prepare_litellm_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """准备LiteLLM请求参数"""
        # 转换消息格式
        messages = [
            {"role": msg.role, "content": msg.content, **({"name": msg.name} if msg.name else {})}
            for msg in request.messages
        ]
        
        # 基础参数
        llm_request = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream or False,
        }
        
        # 添加可选参数
        optional_params = [
            "temperature", "max_tokens", "top_p", "frequency_penalty",
            "presence_penalty", "stop", "n", "user"
        ]
        
        for param in optional_params:
            value = getattr(request, param, None)
            if value is not None:
                llm_request[param] = value
        
        return llm_request
    
    async def _handle_non_stream_completion(
        self, request_id: str, llm_request: Dict[str, Any],
        context: RequestContext, metrics
    ) -> ChatCompletionResponse:
        """处理非流式补全"""
        
        try:
            # 记录下游请求时间
            downstream_start = time.time()
            
            # 调用LiteLLM
            response = await acompletion(**llm_request)
            
            downstream_time = time.time() - downstream_start
            
            # 记录下游响应
            response_data = {
                "id": response.id,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }
            
            # 完成交互记录
            llm_interaction_logger.complete_interaction(
                request_id, response_data, downstream_time, success=True
            )
            
            # 获取token使用情况用于指标
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            # 完成指标记录
            metrics_collector.complete_request(
                request_id,
                success=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
            # 转换为我们的响应格式
            choices = [
                ChatCompletionChoice(
                    index=choice.index,
                    message=ChatMessage(
                        role=choice.message.role,
                        content=choice.message.content or ""
                    ),
                    finish_reason=choice.finish_reason
                )
                for choice in response.choices
            ]
            
            usage = None
            if response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            
            return ChatCompletionResponse(
                id=response.id,
                model=response.model,
                choices=choices,
                usage=usage
            )
        
        except Exception as e:
            llm_interaction_logger.log_error_interaction(request_id, e, "non_stream_completion")
            raise e
    
    async def _handle_stream_completion(
        self, request_id: str, llm_request: Dict[str, Any],
        context: RequestContext, metrics
    ) -> AsyncGenerator[str, None]:
        """处理流式补全"""
        
        accumulated_content = ""
        chunk_count = 0
        finish_reason = None
        prompt_tokens = 0
        completion_tokens = 0
        
        try:
            # 记录下游请求时间
            downstream_start = time.time()
            
            # 调用LiteLLM流式API
            response_stream = await acompletion(**llm_request)
            
            async for chunk in response_stream:
                chunk_count += 1
                
                if chunk.choices:
                    choice = chunk.choices[0]
                    
                    # 累积内容
                    if choice.delta.content:
                        accumulated_content += choice.delta.content
                        
                        # 流式块记录已集成到最终响应中
                    
                    # 记录完成原因
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    
                    # 构建响应块
                    delta = DeltaMessage()
                    if choice.delta.role:
                        delta.role = choice.delta.role
                    if choice.delta.content:
                        delta.content = choice.delta.content
                    
                    chunk_choice = ChatCompletionChunkChoice(
                        index=choice.index,
                        delta=delta,
                        finish_reason=choice.finish_reason
                    )
                    
                    response_chunk = ChatCompletionChunk(
                        id=chunk.id,
                        model=chunk.model,
                        choices=[chunk_choice]
                    )
                    
                    yield f"data: {response_chunk.model_dump_json()}\n\n"
            
            downstream_time = time.time() - downstream_start
            
            # 估算token使用（LiteLLM流式可能没有usage）
            if accumulated_content:
                # 简单估算：1 token ≈ 4 字符（英文）或 1.5 字符（中文）
                completion_tokens = max(1, len(accumulated_content) // 3)
            
            # 估算输入token（基于请求消息长度）
            if prompt_tokens == 0:
                total_input_text = " ".join([msg["content"] for msg in llm_request["messages"]])
                prompt_tokens = max(1, len(total_input_text) // 3)  # 使用相同的估算方法
            
            # 记录下游响应摘要
            response_summary = {
                "content": accumulated_content,
                "finish_reason": finish_reason,
                "chunk_count": chunk_count,
                "estimated_tokens": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
            # 完成交互记录
            llm_interaction_logger.complete_interaction(
                request_id, response_summary, downstream_time, success=True
            )
            
            # 完成指标记录
            metrics_collector.complete_request(
                request_id,
                success=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
            # 发送结束标记
            yield "data: [DONE]\n\n"
        
        except Exception as e:
            llm_interaction_logger.log_error_interaction(request_id, e, "stream_completion")
            
            # 完成指标记录
            metrics_collector.complete_request(
                request_id, success=False, error_message=str(e)
            )
            
            # 在流式响应中发送错误
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        # LiteLLM支持的常见模型
        common_models = [
            # OpenAI
            "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
            # Anthropic
            "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
            # Google
            "gemini-pro", "gemini-pro-vision",
            # Mistral
            "mistral-small", "mistral-medium", "mistral-large",
            # 其他
            "command-nightly", "llama-2-70b-chat"
        ]
        
        return common_models


# 全局LLM服务实例
llm_service = LLMService()