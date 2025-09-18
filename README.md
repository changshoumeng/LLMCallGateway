# LLMCallGateway

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a693.svg)
![LiteLLM](https://img.shields.io/badge/LiteLLM-1.17+-purple.svg)
![Loguru](https://img.shields.io/badge/logging-loguru-orange.svg)

🚀 **专业 LLM API 网关服务** - 统一多模型格式、详细交互日志、完整性能监控

LLMCallGateway 是一个基于 LiteLLM 构建的专业 LLM API 网关服务。将所有模型（包括非 OpenAI 模型）的请求格式统一为 OpenAI 格式，并提供详细的日志跟踪和性能监控，帮助开发者直观了解与下游 LLM API 交互的细节与成本。

## 📖 目录

- [✨ 特性](#-特性)
- [🌟 新增功能](#-新增功能)
- [🚀 快速开始](#-快速开始)
  - [📋 前置要求](#-前置要求)
  - [🛠️ 安装步骤](#️-安装步骤)
  - [⚙️ 环境配置](#️-环境配置)
  - [🏃‍♂️ 启动服务](#️-启动服务)
- [📚 详细文档](#-详细文档)
  - [🌐 API 文档](#-api-文档)
  - [💬 使用示例](#-使用示例)
  - [🏗️ 项目结构](#️-项目结构)
- [🔧 开发](#-开发)
- [🚀 部署](#-部署)
- [🔒 安全性](#-安全性)
- [🤝 贡献](#-贡献)
- [📄 许可证](#-许可证)
- [📞 支持](#-支持)

## ✨ 特性

### 🎯 核心功能
- 🔄 **多模型统一**: 基于 LiteLLM，支持 100+ 模型统一为 OpenAI 格式
- 📊 **详细日志跟踪**: 专门的 LLM 交互日志，完整记录请求/响应细节
- 📈 **性能监控**: 实时统计 token 使用量、处理时延、成功率等关键指标
- 🏗️ **模块化架构**: 专业的项目结构，易于维护和扩展

### ⚡ 技术特性
- 🔄 **完全兼容**: 100% 兼容 OpenAI API 格式，支持聊天补全和模型列表
- 📡 **流式支持**: 支持流式和非流式响应，实时交互体验
- ⚡ **高性能**: 基于 FastAPI + Uvicorn，异步处理，高并发支持
- 🌐 **CORS 支持**: 开箱即用的跨域支持，前端可直接调用
- 🔒 **安全优先**: 环境变量管理，防止敏感信息泄露
- 🐳 **容器化**: 提供 Docker 支持，一键部署
- 🔧 **开发友好**: 热重载、详细错误提示、完整的开发工具链

## 🌟 新增功能

### 📊 指标监控
- **实时指标**: `/metrics` - 获取当前服务指标
- **模型统计**: `/metrics/models` - 按模型分组的使用统计
- **趋势分析**: `/metrics/trends` - 历史趋势数据

### 📝 专业日志
- **系统日志**: `logs/llmcallgateway_system_*.log` - 系统运行日志
- **LLM 交互日志**: `logs/llmcallgateway_llm_interactions_*.log` - 详细的 LLM 交互记录
- **错误日志**: `logs/llmcallgateway_error_*.log` - 错误和异常日志

### 🔍 支持的模型
基于 LiteLLM，支持多种模型提供商：
- **OpenAI**: gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo
- **Anthropic**: claude-3-sonnet, claude-3-haiku
- **Google**: gemini-pro, gemini-pro-vision
- **Mistral**: mistral-small, mistral-medium, mistral-large
- **其他**: command-nightly, llama-2-70b-chat 等

## 🚀 快速开始

### 📋 前置要求

- Python 3.8 或更高版本
- pip 包管理器
- 有效的 LLM API 密钥（OpenAI、Anthropic、Google 等）

### 🛠️ 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/changshoumeng/LLMCallGateway.git
   cd LLMCallGateway
   ```

2. **创建虚拟环境**（推荐）
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或者
   venv\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### ⚙️ 环境配置

1. **复制环境变量模板**
   ```bash
   cp .env.example .env
   ```

2. **编辑配置文件**
   ```bash
   nano .env  # 或使用您喜欢的编辑器
   ```

3. **配置必需参数**
   ```bash
   # 🔑 必需配置（LiteLLM 格式）
   LITELLM_API_KEY=your_api_key_here
   LITELLM_BASE_URL=https://api.openai.com/v1
   
   # 🔄 向后兼容（旧格式仍然支持）
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_BASE_URL=https://api.openai.com/v1
   
   # ⚙️ 可选配置
   HOST=0.0.0.0
   PORT=8728
   ENABLE_RELOAD=false
   DEBUG=false
   LOG_LEVEL=INFO
   ```

> ⚠️ **安全提醒**: 请确保 `.env` 文件不会被提交到版本控制系统！

### 🏃‍♂️ 启动服务

```bash
python main.py
```

🎉 服务将在 `http://localhost:8728` 启动！

## 📚 详细文档

### 🌐 API 文档

启动服务后，您可以访问以下地址查看完整的 API 文档：

- **Swagger UI**: http://localhost:8728/docs
- **ReDoc**: http://localhost:8728/redoc

### 💬 使用示例

#### Python 客户端示例

```python
from openai import OpenAI

# 连接到 LLMCallGateway 服务
client = OpenAI(
    api_key="your_api_key",  # 使用您在 .env 中配置的密钥
    base_url="http://localhost:8728/v1"
)

# 非流式聊天
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "周树人和鲁迅是什么关系？"}
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

#### cURL 示例

```bash
# 非流式请求
curl -X POST "http://localhost:8728/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
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
```

#### 指标查询示例

```bash
# 获取实时指标
curl http://localhost:8728/metrics

# 获取模型统计
curl http://localhost:8728/metrics/models

# 获取24小时趋势
curl http://localhost:8728/metrics/trends?hours=24
```

## 🏗️ 项目结构

```
LLMCallGateway/
├── main.py                 # 主应用入口
├── app/                    # 应用模块
│   ├── core/              # 核心模块
│   │   ├── config.py      # 配置管理
│   │   └── logging.py     # 日志系统
│   ├── services/          # 服务层
│   │   ├── llm_service.py # LLM 代理服务
│   │   └── metrics.py     # 指标统计服务
│   ├── models/            # 数据模型
│   │   └── api_models.py  # API 数据模型
│   └── utils/             # 工具函数
│       └── helpers.py     # 辅助函数
├── logs/                  # 日志目录
├── requirements.txt       # 依赖列表
├── .env.example          # 环境变量模板
└── README.md             # 项目文档
```

## 🔧 开发

### 开发环境搭建

1. **启用开发模式**
   ```bash
   # 在 .env 文件中设置
   ENABLE_RELOAD=true
   DEBUG=true
   ```

2. **运行测试客户端**
   ```bash
   python client.py
   ```

### 专业日志系统

LLMCallGateway 提供分层的专业日志系统：

#### 📝 日志特性
- **分离式日志**: 系统日志与 LLM 交互日志分别存储
- **请求追踪**: 每个请求分配唯一 ID，便于追踪完整生命周期
- **详细记录**: 完整记录用户 Prompt 和 LLM 响应内容
- **下游跟踪**: 详细记录与下游 LLM API 的完整交互过程
- **Token 统计**: 精确统计输入、输出和总 Token 消耗
- **性能监控**: 记录处理时延、成功率等关键指标
- **智能轮转**: 自动日志轮转和清理
- **UTF-8 支持**: 完美支持中文字符

#### 📊 LLM 交互日志示例
```
2024-09-18 13:05:22.123 | LLM | [a1b2c3d4] ▶️ REQUEST START | Model: gpt-4o-mini | User: 你好，请介绍一下自己
2024-09-18 13:05:22.125 | LLM | [a1b2c3d4] ⬇️ DOWNSTREAM REQUEST | Provider: litellm
2024-09-18 13:05:22.126 | LLM | [a1b2c3d4] 📤 Request Data: {"model": "gpt-4o-mini", "messages": [...]}
2024-09-18 13:05:23.456 | LLM | [a1b2c3d4] ⬆️ DOWNSTREAM RESPONSE | Provider: litellm | Time: 1.330s
2024-09-18 13:05:23.457 | LLM | [a1b2c3d4] 📥 Response Data: {"id": "chatcmpl-xyz", "choices": [...]}
2024-09-18 13:05:23.458 | LLM | [a1b2c3d4] ✅ RESPONSE COMPLETE | Status: SUCCESS | Time: 1.335s
2024-09-18 13:05:23.459 | LLM | [a1b2c3d4] 🤖 AI Response: 你好！我是一个AI助手...
2024-09-18 13:05:23.460 | LLM | [a1b2c3d4] 📊 Token Usage: Input=15, Output=45, Total=60
```

## 🚀 部署

### Docker 部署

1. **构建镜像**
   ```bash
   docker build -t llmcallgateway .
   ```

2. **运行容器**
   ```bash
   docker run -p 8728:8728 --env-file .env llmcallgateway
   ```

### Docker Compose

```yaml
version: '3.8'
services:
  llmcallgateway:
    build: .
    ports:
      - "8728:8728"
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

### 生产环境部署

使用 Gunicorn + Nginx 的推荐配置：

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动服务
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8728
```

## 🔒 安全性

### 环境变量安全
- ✅ 所有敏感配置通过环境变量管理
- ✅ `.env` 文件已在 `.gitignore` 中排除
- ✅ 提供 `.env.example` 模板供参考
- ✅ 强制验证必需的环境变量

### 生产环境建议
- 🔐 启用 HTTPS
- 🛡️ 配置反向代理（Nginx/Apache）
- 🔑 定期轮换 API 密钥
- 📊 监控日志文件防止信息泄露
- 🚫 限制 CORS 域名（生产环境）

## 🤝 贡献

我们欢迎所有形式的贡献！

### 💡 报告问题
在 [Issues](https://github.com/changshoumeng/LLMCallGateway/issues) 页面报告 Bug 或提出功能建议。

### 🔧 提交代码
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

### 📝 开发规范
- 遵循 Python PEP 8 代码规范
- 添加适当的类型注解
- 为新功能编写测试
- 更新相关文档

## 📄 许可证

本项目使用 [MIT 许可证](LICENSE) - 查看 LICENSE 文件了解详情。

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代、快速的 Web 框架
- [LiteLLM](https://github.com/BerriAI/litellm) - 统一 LLM API 接口
- [Loguru](https://github.com/Delgan/loguru) - 优雅的 Python 日志库
- [OpenAI](https://openai.com/) - API 规范和设计灵感

## 📞 支持

如果您觉得这个项目有用，请给我们一个 ⭐️！

- 📧 Email: [changshoumeng@example.com](mailto:changshoumeng@example.com)
- 🐛 Issues: [GitHub Issues](https://github.com/changshoumeng/LLMCallGateway/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/changshoumeng/LLMCallGateway/discussions)

---

<div align="center">
  <p>用 ❤️ 制作 by <a href="https://github.com/changshoumeng">changshoumeng</a></p>
  <p>如果这个项目对您有帮助，请考虑给它一个 ⭐️</p>
</div>