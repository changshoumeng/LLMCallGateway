"""
配置管理模块
统一管理所有环境变量和应用配置
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Settings(BaseModel):
    """应用配置"""
    
    # 基础配置
    app_name: str = Field(default="LLMCallGateway", description="应用名称")
    app_version: str = Field(default="2.0.0", description="应用版本")
    debug: bool = Field(default=False, description="调试模式")
    
    # 服务器配置
    host: str = Field(default="0.0.0.0", description="服务器主机")
    port: int = Field(default=8728, description="服务器端口")
    reload: bool = Field(default=False, description="热重载")
    
    # LiteLLM配置
    litellm_api_key: Optional[str] = Field(default=None, description="LiteLLM API密钥")
    litellm_base_url: Optional[str] = Field(default=None, description="LiteLLM基础URL")
    
    # OpenAI配置（向后兼容）
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API密钥")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI基础URL")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_dir: str = Field(default="logs", description="日志目录")
    log_rotation: str = Field(default="10 MB", description="日志轮转大小")
    log_retention: int = Field(default=7, description="日志保留天数")
    
    # 中间件配置
    cors_origins: list = Field(default=["*"], description="CORS允许的源")
    enable_cors: bool = Field(default=True, description="启用CORS")
    
    # 性能配置
    request_timeout: int = Field(default=120, description="请求超时时间(秒)")
    max_tokens_limit: int = Field(default=4096, description="最大token限制")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_required_env(key: str) -> str:
    """获取必需的环境变量"""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def load_settings() -> Settings:
    """加载配置"""
    return Settings(
        # 基础配置
        debug=os.getenv("DEBUG", "false").lower() == "true",
        
        # 服务器配置
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8728")),
        reload=os.getenv("ENABLE_RELOAD", "false").lower() == "true",
        
        # LiteLLM配置
        litellm_api_key=os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        litellm_base_url=os.getenv("LITELLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        
        # OpenAI配置（向后兼容）
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        
        # 日志配置
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_dir=os.getenv("LOG_DIR", "logs"),
        log_rotation=os.getenv("LOG_ROTATION", "10 MB"),
        log_retention=int(os.getenv("LOG_RETENTION", "7")),
        
        # 性能配置
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "120")),
        max_tokens_limit=int(os.getenv("MAX_TOKENS_LIMIT", "4096")),
    )


# 全局配置实例
settings = load_settings()