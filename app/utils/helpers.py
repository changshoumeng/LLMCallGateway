"""
工具函数模块
提供通用的辅助函数
"""

import time
import uuid
from typing import Any, Dict, Optional
from fastapi import HTTPException, Request


def generate_request_id() -> str:
    """生成唯一请求ID"""
    return str(uuid.uuid4())[:8]


def get_current_timestamp() -> int:
    """获取当前时间戳"""
    return int(time.time())


def extract_user_id_from_request(request: Request) -> Optional[str]:
    """从请求中提取用户ID"""
    # 尝试从Authorization header中提取
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        # 这里可以根据实际需求解析token获取用户ID
        # 简单起见，直接返回token的一部分作为用户ID
        token = auth_header[7:]  # 移除"Bearer "
        return token[:16] if len(token) > 16 else token
    
    # 尝试从自定义header中提取
    user_id = request.headers.get("x-user-id")
    if user_id:
        return user_id
    
    # 尝试从查询参数中提取
    user_id = request.query_params.get("user_id")
    if user_id:
        return user_id
    
    return None


def validate_model_name(model: str) -> bool:
    """验证模型名称是否有效"""
    if not model or not isinstance(model, str):
        return False
    
    # 基本的模型名称验证
    if len(model) < 1 or len(model) > 100:
        return False
    
    # 不允许包含特殊字符
    invalid_chars = [" ", "\n", "\r", "\t"]
    for char in invalid_chars:
        if char in model:
            return False
    
    return True


def create_error_response(message: str, error_type: str = "error", 
                         status_code: int = 400) -> HTTPException:
    """创建标准错误响应"""
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": error_type,
                "code": status_code
            }
        }
    )


def sanitize_log_content(content: str, max_length: int = 1000) -> str:
    """清理日志内容，限制长度并移除敏感信息"""
    if not content:
        return ""
    
    # 限制长度
    if len(content) > max_length:
        content = content[:max_length] + "..."
    
    # 这里可以添加更多的敏感信息过滤逻辑
    # 例如移除API密钥、密码等
    sensitive_patterns = [
        "api_key", "password", "token", "secret", "key"
    ]
    
    # 简单的敏感信息掩码（实际项目中应该使用更复杂的正则表达式）
    for pattern in sensitive_patterns:
        if pattern in content.lower():
            # 这里只是示例，实际应该用正则表达式精确匹配和替换
            pass
    
    return content


def format_duration(seconds: float) -> str:
    """格式化持续时间"""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def safe_json_serialize(obj: Any) -> Dict[str, Any]:
    """安全的JSON序列化，处理不可序列化的对象"""
    try:
        import json
        # 尝试序列化，如果失败则转换为字符串
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if hasattr(obj, '__dict__'):
            return {k: str(v) for k, v in obj.__dict__.items()}
        else:
            return {"value": str(obj), "type": type(obj).__name__}


def calculate_tokens_estimate(text: str) -> int:
    """估算文本的token数量"""
    if not text:
        return 0
    
    # 简单的token估算算法
    # 英文：大约4个字符 = 1个token
    # 中文：大约1.5个字符 = 1个token
    
    # 统计中文字符数
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    
    # 统计总字符数
    total_chars = len(text)
    
    # 英文字符数
    english_chars = total_chars - chinese_chars
    
    # 估算token数
    estimated_tokens = (chinese_chars / 1.5) + (english_chars / 4.0)
    
    return max(1, int(estimated_tokens))


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本并添加后缀"""
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix