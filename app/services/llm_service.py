"""
LLM代理服务
基于LiteLLM实现多模型统一接口，支持详细的交互日志和指标统计
"""

import uuid
import time
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator, Union, Tuple
import json

import litellm
from litellm import completion, acompletion, embedding, aembedding

from ..core.config import settings
from ..core.logging import system_logger, llm_interaction_logger
from ..models.api_models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk,
    ChatMessage, ChatCompletionChoice, ChatCompletionChunkChoice,
    DeltaMessage, Usage, RequestContext,
    EmbeddingRequest, EmbeddingResponse, EmbeddingData,
    ToolCall, ToolCallFunction, ToolCallDelta
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
    
    def _normalize_tool_call_entry(self, tool_call: Any) -> Optional[Dict[str, Any]]:
        """将工具调用对象标准化为dict"""
        try:
            if hasattr(tool_call, "model_dump"):
                data = tool_call.model_dump(exclude_none=True)
            elif isinstance(tool_call, dict):
                data = {k: v for k, v in tool_call.items() if v is not None}
            else:
                data = {
                    "id": getattr(tool_call, "id", None),
                    "type": getattr(tool_call, "type", None),
                }
                func = getattr(tool_call, "function", None)
                if func is not None:
                    if hasattr(func, "model_dump"):
                        data["function"] = func.model_dump(exclude_none=True)
                    elif isinstance(func, dict):
                        data["function"] = {k: v for k, v in func.items() if v is not None}
                    else:
                        data["function"] = {
                            "name": getattr(func, "name", None),
                            "arguments": getattr(func, "arguments", None)
                        }
                elif isinstance(tool_call, dict) and isinstance(tool_call.get("function"), dict):
                    data["function"] = tool_call["function"]
            call_id = data.get("id")
            func_info = data.get("function", {})
            func_name = func_info.get("name")
            func_args = func_info.get("arguments")
            if call_id and func_name is not None:
                if func_args is None:
                    func_info["arguments"] = ""
                else:
                    func_info["arguments"] = str(func_args)
                data["function"] = func_info
                data.setdefault("type", "function")
                return {
                    "id": str(call_id),
                    "type": str(data.get("type", "function")),
                    "function": {
                        "name": str(func_info.get("name")),
                        "arguments": str(func_info.get("arguments", ""))
                    }
                }
        except Exception:
            return None
        return None

    def _extract_tool_calls(
        self, message: Any
    ) -> Tuple[Optional[List[ToolCall]], Optional[List[Dict[str, Any]]]]:
        """解析工具调用信息，返回模型对象和日志专用数据"""
        tool_calls_attr = getattr(message, "tool_calls", None)
        if not tool_calls_attr:
            return None, None

        tool_call_models: List[ToolCall] = []
        log_tool_calls: List[Dict[str, Any]] = []

        for tool_call in tool_calls_attr:
            normalized = self._normalize_tool_call_entry(tool_call)
            if not normalized:
                continue
            log_tool_calls.append(normalized)
            try:
                tool_call_models.append(
                    ToolCall(
                        id=normalized["id"],
                        type=normalized["type"],
                        function=ToolCallFunction(
                            name=normalized["function"]["name"],
                            arguments=normalized["function"]["arguments"]
                        )
                    )
                )
            except Exception:
                continue

        return (
            tool_call_models if tool_call_models else None,
            log_tool_calls if log_tool_calls else None
        )

    def _extract_function_call_payload(self, message: Any) -> Optional[Dict[str, Any]]:
        """解析旧版function_call字段"""
        fc = getattr(message, "function_call", None)
        if not fc:
            return None
        if hasattr(fc, "model_dump"):
            payload = fc.model_dump(exclude_none=True)
        elif isinstance(fc, dict):
            payload = {k: v for k, v in fc.items() if v is not None}
        else:
            payload = {
                "name": getattr(fc, "name", None),
                "arguments": getattr(fc, "arguments", None)
            }
        if not payload.get("name"):
            return None
        if payload.get("arguments") is None:
            payload["arguments"] = ""
        else:
            payload["arguments"] = str(payload["arguments"])
        return payload

    def _normalize_message_content(self, content: Optional[Union[str, List[Dict[str, Any]]]]) -> str:
        """将消息内容统一转换为字符串，兼容多模态内容块"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    # 优先提取文本字段
                    if "text" in block and block["text"] is not None:
                        fragments.append(str(block["text"]))
                    elif block.get("type") in {"input_text", "output_text"} and block.get("text") is not None:
                        fragments.append(str(block["text"]))
                    elif block.get("type") == "image_url" and isinstance(block.get("image_url"), dict):
                        image_url = block["image_url"].get("url")
                        if image_url:
                            fragments.append(f"[image:{image_url}]")
                    else:
                        try:
                            fragments.append(json.dumps(block, ensure_ascii=False))
                        except TypeError:
                            fragments.append(str(block))
                else:
                    fragments.append(str(block))
            return "\n".join([frag for frag in fragments if frag]).strip()
        return str(content)
    
    def _extract_user_query(self, messages: List[ChatMessage]) -> str:
        """提取用户查询内容"""
        user_messages: List[str] = []
        for msg in messages:
            if msg.role != "user":
                continue
            normalized = self._normalize_message_content(msg.content)
            if normalized:
                user_messages.append(normalized)
        if user_messages:
            return user_messages[-1]  # 返回最后一条用户消息
        return "无用户查询"
    
    def _prepare_litellm_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        准备发送给LiteLLM的请求参数。

        将ChatCompletionRequest对象转换为下游兼容格式，并保留OpenAI API的扩展字段，如
        工具定义（tools/functions）以及调用策略（tool_choice/function_call）。
        """
        # 转换消息格式: role、content、name
        messages: List[Dict[str, Any]] = []
        for msg in request.messages:
            m: Dict[str, Any] = {"role": msg.role}
            # 处理content字段：允许字符串或内容块列表
            if msg.content is not None:
                m["content"] = msg.content
            elif msg.tool_calls or msg.function_call:
                # 当存在工具/函数调用时允许content为None以遵循OpenAI规范
                m["content"] = None
            else:
                m["content"] = ""
            if msg.name:
                m["name"] = msg.name
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.function_call:
                m["function_call"] = msg.function_call
            if msg.tool_calls:
                m["tool_calls"] = [
                    tool_call.model_dump(exclude_none=True)
                    if hasattr(tool_call, "model_dump")
                    else tool_call  # 允许直接传入字典
                    for tool_call in msg.tool_calls
                ]
            messages.append(m)

        llm_request: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": bool(request.stream),
        }

        # 通用可选参数
        optional_params = [
            "temperature", "max_tokens", "top_p", "frequency_penalty",
            "presence_penalty", "stop", "n", "user"
        ]
        for param in optional_params:
            value = getattr(request, param, None)
            if value is not None:
                llm_request[param] = value

        # 处理工具/函数定义
        if request.tools is not None:
            llm_request["tools"] = request.tools
        elif request.functions is not None:
            # 兼容旧版函数定义
            llm_request["functions"] = request.functions

        # 处理调用策略
        if request.tool_choice is not None:
            llm_request["tool_choice"] = request.tool_choice
        elif request.function_call is not None:
            llm_request["function_call"] = request.function_call

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
            
            # 转换为我们的响应格式
            converted_choices: List[ChatCompletionChoice] = []
            for list_index, choice in enumerate(response.choices):
                # 解析工具调用（如果有）
                tool_calls_list, log_tool_calls = self._extract_tool_calls(choice.message)
                function_call_payload = self._extract_function_call_payload(choice.message)

                # 构建日志用message
                message_payload: Dict[str, Any] = {
                    "role": choice.message.role,
                    "content": choice.message.content
                }
                if getattr(choice.message, "tool_call_id", None):
                    message_payload["tool_call_id"] = choice.message.tool_call_id
                if log_tool_calls:
                    message_payload["tool_calls"] = log_tool_calls
                if function_call_payload:
                    message_payload["function_call"] = function_call_payload

                # 更新日志数据choices
                if list_index < len(response_data["choices"]):
                    response_data["choices"][list_index]["message"] = message_payload

                # 构建ChatMessage
                msg_content = choice.message.content if choice.message.content is not None else None
                tool_call_id = getattr(choice.message, "tool_call_id", None)
                chat_msg = ChatMessage(
                    role=choice.message.role,
                    content=msg_content,
                    tool_call_id=tool_call_id,
                    tool_calls=tool_calls_list,
                    function_call=function_call_payload
                )
                converted_choices.append(
                    ChatCompletionChoice(
                        index=choice.index,
                        message=chat_msg,
                        finish_reason=choice.finish_reason
                    )
                )

            choices = converted_choices
            
            usage = None
            if response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )

            # 完成交互记录（包含工具调用信息）
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
        tool_call_accumulator: Dict[str, Dict[str, Any]] = {}
        function_call_accumulator: Optional[Dict[str, Any]] = None
        
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
                    # 设置角色
                    if getattr(choice.delta, "role", None):
                        delta.role = choice.delta.role
                    # 设置内容
                    if getattr(choice.delta, "content", None):
                        delta.content = choice.delta.content
                    # 解析工具调用增量
                    tool_calls_delta: Optional[List[ToolCallDelta]] = None
                    # 兼容旧版function_call增量
                    if getattr(choice.delta, "function_call", None):
                        fc_delta = choice.delta.function_call
                        fc_payload: Optional[Dict[str, Any]] = None
                        if hasattr(fc_delta, "model_dump"):
                            fc_payload = fc_delta.model_dump(exclude_none=True)
                        elif isinstance(fc_delta, dict):
                            fc_payload = fc_delta
                        else:
                            try:
                                fc_payload = dict(fc_delta)
                            except Exception:
                                fc_payload = None
                        if fc_payload:
                            # 累积function_call信息
                            if function_call_accumulator is None:
                                function_call_accumulator = {
                                    "name": fc_payload.get("name"),
                                    "arguments": ""
                                }
                            if fc_payload.get("name"):
                                function_call_accumulator["name"] = fc_payload["name"]
                            if fc_payload.get("arguments"):
                                function_call_accumulator["arguments"] = (
                                    function_call_accumulator.get("arguments", "") + fc_payload["arguments"]
                                )
                            delta.function_call = fc_payload
                    if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                        tool_calls_delta = []
                        for tc in choice.delta.tool_calls:
                            try:
                                tc_id = getattr(tc, "id", None)
                                if tc_id is None and isinstance(tc, dict):
                                    tc_id = tc.get("id")
                                tc_type = getattr(tc, "type", None)
                                if tc_type is None and isinstance(tc, dict):
                                    tc_type = tc.get("type")
                                # 函数信息
                                if hasattr(tc, "function"):
                                    func = tc.function
                                    func_name = getattr(func, "name", None)
                                    func_args = getattr(func, "arguments", None)
                                elif isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                                    func_name = tc["function"].get("name")
                                    func_args = tc["function"].get("arguments")
                                else:
                                    func_name = None
                                    func_args = None
                                # 构建增量对象
                                if func_name is not None and func_args is not None:
                                    tool_calls_delta.append(
                                        ToolCallDelta(
                                            id=tc_id,
                                            type=tc_type,
                                            function=ToolCallFunction(
                                                name=str(func_name), arguments=str(func_args)
                                            )
                                        )
                                    )
                                    # 累积工具调用信息
                                    call_id = str(tc_id) if tc_id else None
                                    if call_id:
                                        existing = tool_call_accumulator.get(call_id, {
                                            "id": call_id,
                                            "type": str(tc_type) if tc_type else "function",
                                            "function": {"name": None, "arguments": ""}
                                        })
                                        if func_name:
                                            existing["function"]["name"] = str(func_name)
                                        if func_args:
                                            prev_args = existing["function"].get("arguments") or ""
                                            existing["function"]["arguments"] = prev_args + str(func_args)
                                        tool_call_accumulator[call_id] = existing
                            except Exception:
                                continue
                    if tool_calls_delta:
                        delta.tool_calls = tool_calls_delta
                    
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
                total_input_text = " ".join([msg["content"] for msg in llm_request["messages"] if msg.get("content")])
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

            if tool_call_accumulator:
                response_summary["tool_calls"] = [
                    {
                        "id": call_id,
                        "type": data.get("type", "function"),
                        "function": {
                            "name": data.get("function", {}).get("name"),
                            "arguments": data.get("function", {}).get("arguments", "")
                        }
                    }
                    for call_id, data in tool_call_accumulator.items()
                ]
            if function_call_accumulator:
                response_summary["function_call"] = function_call_accumulator
            
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

    def _try_decode_tokens(self, input_data) -> Optional[str]:
        """
        尝试将tokenized数组解码为文本
        如果输入是token数组，尝试使用tiktoken解码
        """
        try:
            import tiktoken

            # 检查是否是token数组
            if isinstance(input_data, list) and len(input_data) > 0:
                if all(isinstance(x, int) for x in input_data):
                    # 尝试使用不同的编码器解码
                    encoders = ["cl100k_base", "gpt2", "r50k_base", "p50k_base"]

                    for encoder_name in encoders:
                        try:
                            encoding = tiktoken.get_encoding(encoder_name)
                            decoded_text = encoding.decode(input_data)
                            if decoded_text and len(decoded_text.strip()) > 0:
                                system_logger.info(f"🔄 自动解码token数组 ({encoder_name}): {len(input_data)} tokens -> '{decoded_text[:50]}...'")
                                return decoded_text
                        except Exception:
                            continue

                    # 如果所有编码器都失败，返回None
                    system_logger.warning(f"⚠️ 无法解码token数组: {input_data[:10]}...")
                    return None

            return None
        except ImportError:
            system_logger.warning("⚠️ tiktoken库未安装，无法解码token数组")
            return None
        except Exception as e:
            system_logger.error(f"解码token数组时出错: {e}")
            return None

    def _preprocess_embedding_input(self, request: EmbeddingRequest) -> EmbeddingRequest:
        """
        预处理嵌入输入，支持自动token解码
        """
        processed_input = request.input

        # 处理单个输入
        if isinstance(request.input, list) and len(request.input) > 0:
            # 检查是否是token数组 (所有元素都是int)
            if all(isinstance(x, int) for x in request.input):
                decoded_text = self._try_decode_tokens(request.input)
                if decoded_text:
                    processed_input = decoded_text
                    system_logger.info(f"✅ 成功解码tokenized输入为文本")
                else:
                    # 如果解码失败，抛出友好的错误
                    raise ValueError(
                        "检测到tokenized数字数组但无法解码。请发送原始文本字符串而不是token数组。"
                        f"\n正确格式: '原始文本字符串'"
                        f"\n错误格式: {request.input[:10]}..."
                    )
            # 处理列表中包含token数组的情况
            elif any(isinstance(item, list) and all(isinstance(x, int) for x in item) for item in request.input if isinstance(item, list)):
                processed_list = []
                for item in request.input:
                    if isinstance(item, list) and all(isinstance(x, int) for x in item):
                        decoded_text = self._try_decode_tokens(item)
                        if decoded_text:
                            processed_list.append(decoded_text)
                        else:
                            raise ValueError(f"无法解码token数组: {item[:10]}...")
                    else:
                        processed_list.append(item)
                processed_input = processed_list
                system_logger.info(f"✅ 成功解码列表中的tokenized输入")

        # 创建处理后的请求对象
        return EmbeddingRequest(
            input=processed_input,
            model=request.model,
            encoding_format=request.encoding_format,
            dimensions=request.dimensions,
            user=request.user
        )

    async def create_embeddings(self, request: EmbeddingRequest,
                               user_id: Optional[str] = None) -> EmbeddingResponse:
        """创建文本嵌入"""
        # 预处理输入，支持自动token解码
        try:
            request = self._preprocess_embedding_input(request)
        except ValueError as e:
            system_logger.error(f"输入预处理失败: {e}")
            raise ValueError(str(e))

        # 生成请求ID
        request_id = str(uuid.uuid4())[:8]

        # 创建请求上下文
        context = RequestContext(
            request_id=request_id,
            start_time=time.time(),
            model=request.model,
            user_id=user_id,
            stream=False,  # embeddings不支持流式
            request_type="embedding"
        )

        # 开始记录指标
        metrics = metrics_collector.start_request(context)

        try:
            # 准备LiteLLM请求参数
            llm_request = self._prepare_embedding_request(request)

            # 开始记录下游交互
            llm_interaction_logger.start_interaction(
                request_id, "litellm", llm_request
            )

            # 处理嵌入请求
            return await self._handle_embedding_request(
                request_id, llm_request, context, metrics
            )

        except Exception as e:
            # 记录错误交互
            llm_interaction_logger.log_error_interaction(request_id, e, "create_embeddings")

            # 完成指标记录
            metrics_collector.complete_request(
                request_id, success=False, error_message=str(e)
            )

            system_logger.error(f"嵌入生成失败: {e}")
            raise e

    def _prepare_embedding_request(self, request: EmbeddingRequest) -> Dict[str, Any]:
        """准备嵌入请求参数"""
        llm_request = {
            "model": request.model,
            "input": request.input,
        }

        # 添加可选参数
        if request.user:
            llm_request["user"] = request.user

        if request.dimensions:
            llm_request["dimensions"] = request.dimensions

        if request.encoding_format:
            llm_request["encoding_format"] = request.encoding_format

        return llm_request

    async def _handle_embedding_request(
        self, request_id: str, llm_request: Dict[str, Any],
        context: RequestContext, metrics
    ) -> EmbeddingResponse:
        """处理嵌入请求"""

        try:
            # 记录下游请求时间
            downstream_start = time.time()

            # 调用LiteLLM嵌入API
            response = await aembedding(**llm_request)

            downstream_time = time.time() - downstream_start

            # 处理LiteLLM响应 - 根据实际测试，LiteLLM返回的是对象格式
            # 但 data 字段包含的是字典列表，不是对象列表
            response_data = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": data["embedding"],  # data是字典，使用字典访问
                        "index": data["index"]
                    }
                    for data in response.data
                ],
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }
            model_name = response.model
            usage_info = response.usage if response.usage else {}

            # 完成交互记录
            llm_interaction_logger.complete_interaction(
                request_id, response_data, downstream_time, success=True
            )

            # 获取token使用情况用于指标
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            total_tokens = usage_info.get("total_tokens", 0)

            # 完成指标记录
            metrics_collector.complete_request(
                request_id,
                success=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,  # embeddings没有completion tokens
                total_tokens=total_tokens
            )

            # 转换为我们的响应格式
            # LiteLLM返回对象格式，但data是字典列表
            embedding_data = [
                EmbeddingData(
                    embedding=data["embedding"],  # data是字典，使用字典访问
                    index=data["index"]
                )
                for data in response.data
            ]

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,  # embeddings没有completion tokens
                total_tokens=total_tokens
            )

            return EmbeddingResponse(
                data=embedding_data,
                model=model_name,
                usage=usage
            )

        except Exception as e:
            llm_interaction_logger.log_error_interaction(request_id, e, "embedding_request")
            raise e

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
