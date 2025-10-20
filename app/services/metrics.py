"""
指标统计服务
跟踪和统计API使用情况、性能指标等
"""

import time
from typing import Dict, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock

from ..models.api_models import RequestContext


@dataclass
class RequestMetrics:
    """单次请求指标"""
    request_id: str
    model: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    user_id: Optional[str] = None
    stream: bool = False
    request_type: str = "chat"  # chat, embedding
    
    @property
    def duration(self) -> float:
        """请求处理时长"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.end_time is not None


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._lock = Lock()
        
        # 活跃请求
        self._active_requests: Dict[str, RequestMetrics] = {}
        
        # 已完成请求历史（使用deque限制大小）
        self._completed_requests: deque = deque(maxlen=max_history)
        
        # 聚合统计
        self._total_requests = 0
        self._total_tokens = 0
        self._total_duration = 0.0
        self._success_count = 0
        self._model_usage: Dict[str, int] = defaultdict(int)
        self._hourly_stats: Dict[str, Dict] = defaultdict(lambda: {
            'requests': 0,
            'tokens': 0,
            'duration': 0.0,
            'errors': 0
        })
    
    def start_request(self, context: RequestContext) -> RequestMetrics:
        """开始记录请求"""
        with self._lock:
            metrics = RequestMetrics(
                request_id=context.request_id,
                model=context.model,
                start_time=context.start_time,
                user_id=context.user_id,
                stream=context.stream,
                request_type=context.request_type
            )
            self._active_requests[context.request_id] = metrics
            return metrics
    
    def complete_request(self, request_id: str, success: bool = True,
                        error_message: Optional[str] = None,
                        prompt_tokens: int = 0,
                        completion_tokens: int = 0,
                        total_tokens: Optional[int] = None) -> Optional[RequestMetrics]:
        """完成请求记录"""
        with self._lock:
            if request_id not in self._active_requests:
                return None
            
            metrics = self._active_requests.pop(request_id)
            metrics.end_time = time.time()
            metrics.success = success
            metrics.error_message = error_message
            metrics.prompt_tokens = prompt_tokens
            metrics.completion_tokens = completion_tokens
            # 使用提供的total_tokens，否则计算出来
            metrics.total_tokens = total_tokens if total_tokens is not None else (prompt_tokens + completion_tokens)
            
            # 添加到历史记录
            self._completed_requests.append(metrics)
            
            # 更新聚合统计
            self._update_aggregated_stats(metrics)
            
            return metrics
    
    def _update_aggregated_stats(self, metrics: RequestMetrics) -> None:
        """更新聚合统计（调用时已持有锁）"""
        self._total_requests += 1
        self._total_tokens += metrics.total_tokens
        self._total_duration += metrics.duration
        
        if metrics.success:
            self._success_count += 1
        
        self._model_usage[metrics.model] += 1
        
        # 按小时统计
        hour_key = time.strftime('%Y-%m-%d-%H', time.localtime(metrics.start_time))
        hour_stats = self._hourly_stats[hour_key]
        hour_stats['requests'] += 1
        hour_stats['tokens'] += metrics.total_tokens
        hour_stats['duration'] += metrics.duration
        if not metrics.success:
            hour_stats['errors'] += 1
    
    def get_current_stats(self) -> Dict:
        """获取当前统计数据"""
        with self._lock:
            if self._total_requests == 0:
                return {
                    'total_requests': 0,
                    'total_tokens': 0,
                    'average_latency': 0.0,
                    'success_rate': 1.0,
                    'active_requests': 0,
                    'models_used': {},
                    'requests_per_hour': 0,
                    'tokens_per_hour': 0
                }
            
            current_hour = time.strftime('%Y-%m-%d-%H')
            hour_stats = self._hourly_stats.get(current_hour, {'requests': 0, 'tokens': 0})
            
            return {
                'total_requests': self._total_requests,
                'total_tokens': self._total_tokens,
                'average_latency': self._total_duration / self._total_requests if self._total_requests > 0 else 0.0,
                'success_rate': self._success_count / self._total_requests if self._total_requests > 0 else 1.0,
                'active_requests': len(self._active_requests),
                'models_used': dict(self._model_usage),
                'requests_per_hour': hour_stats['requests'],
                'tokens_per_hour': hour_stats['tokens']
            }
    
    def get_recent_requests(self, limit: int = 100) -> List[RequestMetrics]:
        """获取最近的请求记录"""
        with self._lock:
            # 返回最近的请求（从新到旧）
            recent_requests = list(self._completed_requests)[-limit:]
            recent_requests.reverse()
            return recent_requests
    
    def get_model_stats(self) -> Dict[str, Dict]:
        """获取按模型分组的统计"""
        with self._lock:
            model_stats = {}
            
            for request in self._completed_requests:
                model = request.model
                if model not in model_stats:
                    model_stats[model] = {
                        'requests': 0,
                        'tokens': 0,
                        'total_duration': 0.0,
                        'success_count': 0,
                        'error_count': 0
                    }
                
                stats = model_stats[model]
                stats['requests'] += 1
                stats['tokens'] += request.total_tokens
                stats['total_duration'] += request.duration
                
                if request.success:
                    stats['success_count'] += 1
                else:
                    stats['error_count'] += 1
            
            # 计算平均值
            for model, stats in model_stats.items():
                if stats['requests'] > 0:
                    stats['average_latency'] = stats['total_duration'] / stats['requests']
                    stats['success_rate'] = stats['success_count'] / stats['requests']
                else:
                    stats['average_latency'] = 0.0
                    stats['success_rate'] = 1.0
            
            return model_stats
    
    def get_hourly_trends(self, hours: int = 24) -> Dict[str, List]:
        """获取小时级趋势数据"""
        with self._lock:
            current_time = time.time()
            trends = {
                'hours': [],
                'requests': [],
                'tokens': [],
                'latency': [],
                'errors': []
            }
            
            for i in range(hours):
                hour_time = current_time - (i * 3600)
                hour_key = time.strftime('%Y-%m-%d-%H', time.localtime(hour_time))
                hour_stats = self._hourly_stats.get(hour_key, {
                    'requests': 0,
                    'tokens': 0,
                    'duration': 0.0,
                    'errors': 0
                })
                
                trends['hours'].insert(0, time.strftime('%H:00', time.localtime(hour_time)))
                trends['requests'].insert(0, hour_stats['requests'])
                trends['tokens'].insert(0, hour_stats['tokens'])
                
                # 计算平均延迟
                if hour_stats['requests'] > 0:
                    avg_latency = hour_stats['duration'] / hour_stats['requests']
                else:
                    avg_latency = 0.0
                trends['latency'].insert(0, avg_latency)
                trends['errors'].insert(0, hour_stats['errors'])
            
            return trends
    
    def reset_stats(self) -> None:
        """重置所有统计数据"""
        with self._lock:
            self._active_requests.clear()
            self._completed_requests.clear()
            self._total_requests = 0
            self._total_tokens = 0
            self._total_duration = 0.0
            self._success_count = 0
            self._model_usage.clear()
            self._hourly_stats.clear()


# 全局指标收集器实例
metrics_collector = MetricsCollector()