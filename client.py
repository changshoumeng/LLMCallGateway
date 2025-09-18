import asyncio
import json
import time
import requests
from openai import OpenAI

# 配置
BASE_URL = "http://localhost:8728"
API_BASE_URL = f"{BASE_URL}/v1"

def test_chat_completion():
    """测试非流式聊天补全"""
    print("\n💬 测试非流式聊天补全...")
    try:
        client = OpenAI(
            api_key="test-key",  # 测试用，实际不验证
            base_url=API_BASE_URL
        )

        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "周树人和鲁迅是兄弟吗？请简短回答。"}
            ]
        )
        end_time = time.time()

        print(f"✅ 非流式聊天补全成功 (耗时: {end_time - start_time:.2f}秒)")
        print(f"   响应: {response.choices[0].message.content}")
        print(f"   模型: {response.model}")
        print(f"   Token使用: {response.usage.total_tokens if response.usage else 'N/A'}")
        return True
    except Exception as e:
        print(f"❌ 非流式聊天补全失败: {e}")
        return False
test_chat_completion()