# MyGPT - 极简OpenAI兼容API代理服务

MyGPT是一个轻量级的OpenAI兼容API代理服务，内部转发请求到类似于dmxapi.com这样的云服务。

设计原则：代码极简、功能专注、易于部署。

## 特性

- 🚀 **极简设计**: 单文件实现，代码清晰易懂
- 🔄 **完全兼容**: 支持OpenAI API格式的聊天补全和模型列表
- 📡 **流式支持**: 支持流式和非流式响应模式
- ⚡ **快速部署**: 几分钟内即可启动服务
- 🌐 **CORS支持**: 支持前端直接调用
- 📝 **详细日志**: 完整追踪用户prompt和LLM响应，支持请求ID追踪

## 快速开始

### 1. 安装依赖

```bash
cd mygpt
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件：

```bash
# OpenAI API Configuration
OPENAI_API_KEY=
OPENAI_BASE_URL=https://www.dmxapi.com/v1

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### 3. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动。

## API文档

启动服务后，访问以下地址查看API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API端点

### 1. 健康检查

```bash
GET /
```

### 2. 获取模型列表

```bash
GET /v1/models
```

### 3. 聊天补全

```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": "你好，请介绍一下自己。"
    }
  ],
  "stream": false
}
```

## 使用示例

### Python客户端

```python
from openai import OpenAI

# 使用MyGPT服务
client = OpenAI(
    api_key="any-key",  # 由于是代理服务，这里可以是任意值
    base_url="http://localhost:8000/v1"
)

# 非流式聊天
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "周树人和鲁迅是兄弟吗？"}
    ]
)

print(response.choices[0].message.content)

# 流式聊天
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "请写一首关于春天的诗"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### cURL示例

```bash
# 非流式请求
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "你好，请介绍一下自己。"
      }
    ],
    "stream": false
  }'

# 流式请求
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "请写一首关于春天的诗"
      }
    ],
    "stream": true
  }'
```

## 项目结构

```
mygpt/
├── main.py           # 主应用文件
├── requirements.txt  # 项目依赖
├── .env             # 环境变量配置
├── .gitignore       # Git忽略文件
├── README.md        # 项目文档
 
```

## 日志追踪功能

MyGPT支持详细的请求追踪和日志记录，帮助监控和调试API调用：

### 日志内容

- **请求追踪**: 每个请求分配唯一ID，便于追踪完整生命周期
- **用户Prompt**: 完整记录用户提交的对话内容
- **LLM响应**: 记录模型返回的完整响应内容
- **Token使用**: 统计输入、输出和总Token消耗
- **处理状态**: 记录请求开始、处理中、完成等状态

### 日志文件

- **双重输出**: 同时输出到控制台和日志文件
- **文件轮转**: 主日志文件最大10MB，保留5个备份
- **按日分文件**: 每日生成新的日志文件 `mygpt_YYYY-MM-DD.log`
- **错误分离**: 错误日志单独保存在 `mygpt_error_YYYY-MM-DD.log`
- **UTF-8编码**: 支持中文字符正确显示

### 日志格式示例

```
INFO - [a1b2c3d4] 📥 收到聊天补全请求
INFO - [a1b2c3d4] 🎯 模型: gpt-4o-mini
INFO - [a1b2c3d4] 🌊 流式: False
INFO - [a1b2c3d4] 💬 用户对话:
INFO - [a1b2c3d4]   1. [user]: 你好，请介绍一下自己。
INFO - [a1b2c3d4] 🚀 开始非流式响应处理...
INFO - [a1b2c3d4] 🤖 LLM响应内容: 你好！我是一个AI助手...
INFO - [a1b2c3d4] ✅ 完成原因: stop
INFO - [a1b2c3d4] 📊 Token使用 - 输入: 15, 输出: 45, 总计: 60
INFO - [a1b2c3d4] ✨ 非流式响应处理完成
```
 
## 部署建议

### Docker部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### 生产环境

对于生产环境，建议：

1. 使用Gunicorn或类似的WSGI服务器
2. 配置反向代理（Nginx）
3. 设置适当的日志级别
4. 启用HTTPS

```bash
# 使用Gunicorn
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！ 