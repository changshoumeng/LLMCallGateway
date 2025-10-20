"""
专业日志系统模块
分离系统日志和LLM交互日志，提供详细的追踪能力
"""

import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger

from .config import settings


class LoggerManager:
    """日志管理器"""
    
    def __init__(self):
        self.initialized = False
        self.setup_logger()
    
    def setup_logger(self) -> None:
        """配置loguru日志系统"""
        if self.initialized:
            return
            
        # 创建日志目录
        if not os.path.exists(settings.log_dir):
            os.makedirs(settings.log_dir)
        
        # 移除默认处理器
        logger.remove()
        
        # 基础日志格式
        base_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
        
        # 控制台处理器
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | {message}",
            level=settings.log_level,
            colorize=True,
        )
        
        # 系统主日志文件
        today = datetime.now().strftime('%Y-%m-%d')
        main_log_file = os.path.join(settings.log_dir, f'llmcallgateway_system_{today}.log')
        logger.add(
            main_log_file,
            format=base_format,
            level="INFO",
            rotation=settings.log_rotation,
            retention=f"{settings.log_retention} days",
            encoding="utf-8",
            enqueue=True,
        )
        
        # 错误日志文件
        error_log_file = os.path.join(settings.log_dir, f'llmcallgateway_error_{today}.log')
        logger.add(
            error_log_file,
            format=base_format,
            level="ERROR",
            rotation="5 MB",
            retention=f"{settings.log_retention} days",
            encoding="utf-8",
            enqueue=True,
        )
        
        # LLM交互专用日志文件
        llm_log_file = os.path.join(settings.log_dir, f'llmcallgateway_llm_interactions_{today}.log')
        logger.add(
            llm_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | LLM | {message}",
            level="INFO",
            rotation="20 MB",
            retention=f"{settings.log_retention} days",
            encoding="utf-8",
            enqueue=True,
            filter=lambda record: record["extra"].get("log_type") == "llm_interaction"
        )
        
        self.initialized = True
        logger.info(f"日志系统初始化完成 - 日志目录: {settings.log_dir}")
    
    def get_system_logger(self) -> "logger":
        """获取系统日志记录器"""
        return logger.bind(log_type="system")
    
    def get_llm_logger(self) -> "logger":
        """获取LLM交互日志记录器"""
        return logger.bind(log_type="llm_interaction")


class LLMInteractionLogger:
    """
LLM交互专用日志记录器 - 专注于下游服务商的请求/响应跟踪
输出美观的结构化JSON日志
"""
    
    def __init__(self):
        self.llm_logger = logger.bind(log_type="llm_interaction")
        self.interactions = {}  # 存储请求信息用于最终输出
    
    def start_interaction(self, request_id: str, provider: str, request_data: Dict[str, Any]) -> None:
        """开始一个交互记录"""
        from datetime import datetime
        
        self.interactions[request_id] = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "downstream_request": {
                "provider": provider,
                **request_data
            },
            "downstream_response": None,
            "processing_time": None,
            "status": "pending",
            "error": None
        }

    def _truncate_embedding_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """截断embedding字段值，保留前10个字符以减少日志体积"""
        import copy

        truncated_data = copy.deepcopy(data)

        # 处理embeddings响应格式
        if isinstance(truncated_data, dict) and "data" in truncated_data:
            data_list = truncated_data["data"]
            if isinstance(data_list, list):
                for item in data_list:
                    if isinstance(item, dict) and "embedding" in item:
                        embedding = item["embedding"]
                        if isinstance(embedding, str) and len(embedding) > 10:
                            # 截断base64格式的embedding
                            item["embedding"] = embedding[:10] + "..."
                        elif isinstance(embedding, list) and len(embedding) > 3:
                            # 截断float数组格式的embedding，只保留前3个元素
                            item["embedding"] = embedding[:3] + ["..."]

        return truncated_data

    def complete_interaction(self, request_id: str, response_data: Dict[str, Any], processing_time: float, success: bool = True, error: str = None) -> None:
        """完成一个交互记录并输出JSON"""
        if request_id not in self.interactions:
            return
            
        interaction = self.interactions[request_id]
        interaction["downstream_response"] = self._truncate_embedding_fields(response_data)
        interaction["processing_time"] = processing_time
        interaction["status"] = "success" if success else "error"
        if error:
            interaction["error"] = error
        
        # 输出美观的JSON日志
        import json
        json_log = json.dumps(interaction, ensure_ascii=False, indent=2)
        self.llm_logger.info(json_log)
        
        # 清理已完成的交互
        del self.interactions[request_id]
    
    def log_error_interaction(self, request_id: str, error: Exception, context: str = "") -> None:
        """记录错误交互"""
        if request_id in self.interactions:
            self.complete_interaction(
                request_id, 
                {"error_context": context}, 
                0.0, 
                success=False, 
                error=str(error)
            )
        else:
            # 如果没有开始记录，直接记录错误
            from datetime import datetime
            import json
            
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "status": "error",
                "error": str(error),
                "context": context
            }
            
            json_log = json.dumps(error_log, ensure_ascii=False, indent=2)
            self.llm_logger.error(json_log)


# 全局日志管理器实例
log_manager = LoggerManager()

# 导出日志记录器
system_logger = log_manager.get_system_logger()
llm_interaction_logger = LLMInteractionLogger()

# 为了兼容性，导出logger
__all__ = ["system_logger", "llm_interaction_logger", "logger"]