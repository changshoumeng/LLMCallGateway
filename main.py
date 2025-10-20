#!/usr/bin/env python3
"""
LLMCallGateway - 专业LLM API网关服务

基于LiteLLM构建的多模型统一网关服务，支持：
- 将所有模型请求统一为OpenAI格式
- 详细的LLM交互日志跟踪
- 完整的token统计和性能监控
- 模块化架构，易于扩展和维护
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 导入应用模块
from app.core.config import settings
from app.core.logging import system_logger
from app.models.api_models import (
    ChatCompletionRequest, ModelList, Model, HealthResponse, MetricsResponse,
    EmbeddingRequest, EmbeddingResponse
)
from app.services.llm_service import llm_service
from app.services.metrics import metrics_collector
from app.utils.helpers import extract_user_id_from_request, create_error_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    system_logger.info(f"🚀 启动 {settings.app_name} v{settings.app_version}")
    system_logger.info(f"📊 调试模式: {settings.debug}")
    system_logger.info(f"🌐 服务地址: http://{settings.host}:{settings.port}")
    system_logger.info(f"📚 API文档: http://{settings.host}:{settings.port}/docs")
    
    yield
    
    # 关闭时清理
    system_logger.info(f"⏹️ {settings.app_name} 服务正在关闭...")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    description="专业LLM API网关服务 - 统一多模型为OpenAI格式，提供详细的交互日志和性能监控",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()
    
    # 记录请求信息
    user_id = extract_user_id_from_request(request)
    system_logger.info(
        f"📥 {request.method} {request.url.path} | "
        f"User: {user_id or 'Anonymous'} | "
        f"IP: {request.client.host if request.client else 'Unknown'}"
    )
    
    # 处理请求
    response = await call_next(request)
    
    # 记录响应信息
    process_time = time.time() - start_time
    system_logger.info(
        f"📤 {request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Time: {process_time:.3f}s"
    )
    
    # 添加响应头
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Service-Version"] = settings.app_version
    
    return response


# === 辅助函数 ===

async def preprocess_embedding_data(raw_data: dict) -> dict:
    """
    预处理embeddings输入数据，支持自动token解码
    """
    import json

    processed_data = raw_data.copy()

    if "input" in processed_data:
        input_data = processed_data["input"]

        # 处理单个token数组
        if isinstance(input_data, list) and len(input_data) > 0:
            if all(isinstance(x, int) for x in input_data):
                # 尝试解码token数组
                decoded_text = try_decode_tokens(input_data)
                if decoded_text:
                    processed_data["input"] = decoded_text
                    system_logger.info(f"✅ 自动解码token数组为文本: '{decoded_text[:50]}...'")
                else:
                    raise ValueError(
                        f"检测到tokenized数字数组但无法解码。请发送原始文本字符串而不是token数组。"
                        f"\n正确格式: '原始文本字符串'"
                        f"\n错误格式: {input_data[:10]}..."
                    )
            # 处理包含token数组的列表
            elif any(isinstance(item, list) and all(isinstance(x, int) for x in item) for item in input_data if isinstance(item, list)):
                processed_list = []
                for item in input_data:
                    if isinstance(item, list) and all(isinstance(x, int) for x in item):
                        decoded_text = try_decode_tokens(item)
                        if decoded_text:
                            processed_list.append(decoded_text)
                        else:
                            raise ValueError(f"无法解码token数组: {item[:10]}...")
                    else:
                        processed_list.append(item)
                processed_data["input"] = processed_list
                system_logger.info(f"✅ 自动解码列表中的token数组")

    return processed_data


def try_decode_tokens(input_data) -> Optional[str]:
    """
    尝试将tokenized数组解码为文本
    """
    try:
        import tiktoken

        if isinstance(input_data, list) and len(input_data) > 0:
            if all(isinstance(x, int) for x in input_data):
                # 尝试使用不同的编码器解码
                encoders = ["cl100k_base", "gpt2", "r50k_base", "p50k_base"]

                for encoder_name in encoders:
                    try:
                        encoding = tiktoken.get_encoding(encoder_name)
                        decoded_text = encoding.decode(input_data)
                        if decoded_text and len(decoded_text.strip()) > 0:
                            system_logger.info(f"🔄 使用{encoder_name}解码: {len(input_data)} tokens -> 文本")
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


# === API路由定义 ===

@app.get("/", response_model=HealthResponse)
async def root():
    """健康检查和服务信息"""
    return HealthResponse(
        service=settings.app_name,
        status="running",
        version=settings.app_version,
        description="LLM API网关服务 - 统一多模型为OpenAI格式"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """详细健康检查"""
    return HealthResponse(
        service=settings.app_name,
        status="healthy",
        version=settings.app_version,
        description="所有系统正常运行"
    )


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """获取可用模型列表"""
    try:
        available_models = llm_service.get_available_models()
        
        models = [
            Model(
                id=model_id,
                created=int(time.time()),
                owned_by="llmcallgateway"
            )
            for model_id in available_models
        ]
        
        system_logger.info(f"📋 返回模型列表: {len(models)} 个模型")
        return ModelList(data=models)
    
    except Exception as e:
        system_logger.error(f"获取模型列表失败: {e}")
        raise create_error_response("获取模型列表失败", "models_error", 500)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, http_request: Request):
    """创建聊天补全"""
    try:
        # 提取用户ID
        user_id = extract_user_id_from_request(http_request)
        
        # 基本验证
        if not request.messages:
            raise create_error_response("消息列表不能为空", "invalid_request", 400)
        
        if not request.model:
            raise create_error_response("模型名称不能为空", "invalid_request", 400)
        
        # 调用LLM服务
        result = await llm_service.create_chat_completion(request, user_id)
        
        # 根据响应类型返回
        if request.stream:
            return StreamingResponse(
                result,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
                }
            )
        else:
            return result
    
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"聊天补全处理失败: {e}")
        raise create_error_response(f"请求处理失败: {str(e)}", "completion_error", 500)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(http_request: Request):
    """创建文本嵌入 - 支持自动token解码"""
    try:
        # 提取用户ID
        user_id = extract_user_id_from_request(http_request)

        # 获取原始JSON数据
        raw_data = await http_request.json()

        system_logger.info(f"原始JSON数据: {raw_data}")

        # 智能预处理输入数据
        processed_data = await preprocess_embedding_data(raw_data)

        # 创建验证后的请求对象
        try:
            request = EmbeddingRequest(**processed_data)
        except Exception as e:
            system_logger.error(f"请求验证失败: {e}")
            raise create_error_response(f"请求格式错误: {str(e)}", "invalid_request", 400)

        # 基本验证
        if not request.input:
            raise create_error_response("输入文本不能为空", "invalid_request", 400)

        if not request.model:
            raise create_error_response("模型名称不能为空", "invalid_request", 400)

        # 调用LLM服务
        result = await llm_service.create_embeddings(request, user_id)

        return result

    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"嵌入生成处理失败: {e}")
        raise create_error_response(f"请求处理失败: {str(e)}", "embedding_error", 500)


@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """获取服务指标"""
    try:
        stats = metrics_collector.get_current_stats()
        return stats
    except Exception as e:
        system_logger.error(f"获取指标失败: {e}")
        raise create_error_response("获取指标失败", "metrics_error", 500)


@app.get("/metrics/models")
async def get_model_metrics():
    """获取按模型分组的指标"""
    try:
        model_stats = metrics_collector.get_model_stats()
        return model_stats
    except Exception as e:
        system_logger.error(f"获取模型指标失败: {e}")
        raise create_error_response("获取模型指标失败", "metrics_error", 500)


@app.get("/metrics/trends")
async def get_metrics_trends(hours: int = 24):
    """获取指标趋势数据"""
    try:
        if hours < 1 or hours > 168:  # 最多7天
            hours = 24
        
        trends = metrics_collector.get_hourly_trends(hours)
        return trends
    except Exception as e:
        system_logger.error(f"获取趋势数据失败: {e}")
        raise create_error_response("获取趋势数据失败", "metrics_error", 500)


@app.post("/admin/reset-metrics")
async def reset_metrics():
    """重置指标数据（管理员功能）"""
    try:
        metrics_collector.reset_stats()
        system_logger.info("📊 指标数据已重置")
        return {"message": "指标数据已重置", "timestamp": int(time.time())}
    except Exception as e:
        system_logger.error(f"重置指标失败: {e}")
        raise create_error_response("重置指标失败", "admin_error", 500)


# === 错误处理 ===

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """输入验证异常处理器 - 专门处理422错误"""

    def is_tokenized_input_error(errors):
        """检测是否是tokenized输入错误"""
        for error in errors:
            if ('input' in error.get('loc', []) and
                error.get('type') == 'string_type' and
                isinstance(error.get('input'), list) and
                len(error.get('input', [])) > 0 and
                all(isinstance(x, int) for x in error.get('input', [])[:10])):  # 检查前10个元素
                return True
        return False

    # 检查是否是嵌入接口的错误
    is_embedding_request = request.url.path == "/v1/embeddings"
    errors = exc.errors()

    if is_embedding_request and is_tokenized_input_error(errors):
        # 针对tokenized输入提供专门的错误信息
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": "输入格式错误：检测到tokenized数字数组",
                    "type": "invalid_request_error",
                    "code": "invalid_input_format",
                    "details": "embeddings API需要原始文本字符串，不接受tokenized的数字数组。",
                    "correct_examples": {
                        "single_text": '{"input": "Hello world", "model": "text-embedding-3-small"}',
                        "multiple_texts": '{"input": ["Hello", "World"], "model": "text-embedding-3-small"}'
                    },
                    "common_mistakes": [
                        "❌ 不要发送: {\"input\": [3134, 419, 57086], ...}",
                        "❌ 不要发送: {\"input\": [[3134, 419]], ...}",
                        "✅ 应该发送: {\"input\": \"原始文本字符串\", ...}"
                    ]
                }
            }
        )

    # 通用验证错误处理
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "请求格式验证失败",
                "type": "validation_error",
                "details": [
                    {
                        "loc": error["loc"],
                        "msg": error["msg"],
                        "type": error["type"]
                    }
                    for error in errors
                ]
            }
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {"error": {"message": exc.detail}}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    system_logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "内部服务器错误",
                "type": "internal_error",
                "code": 500
            }
        }
    )


# === 应用启动 ===

def main():
    """主函数"""
    system_logger.info(f"🔧 配置信息:")
    system_logger.info(f"   Host: {settings.host}")
    system_logger.info(f"   Port: {settings.port}")
    system_logger.info(f"   Debug: {settings.debug}")
    system_logger.info(f"   Log Level: {settings.log_level}")
    system_logger.info(f"   Reload: {settings.reload}")
    
    if settings.reload:
        system_logger.info("🔄 热重载模式启用")
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=True,
            reload_excludes=["logs/*", "*.log"],
            log_level=settings.log_level.lower()
        )
    else:
        system_logger.info("⚡ 生产模式启动")
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=False,
            log_level=settings.log_level.lower()
        )


if __name__ == "__main__":
    main()