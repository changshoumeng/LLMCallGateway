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
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 导入应用模块
from app.core.config import settings
from app.core.logging import system_logger
from app.models.api_models import (
    ChatCompletionRequest, ModelList, Model, HealthResponse, MetricsResponse
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