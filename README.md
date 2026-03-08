# RAG Memory System - 私有化 AI 记忆系统

> 一个专属于个人的、隐私安全的 RAG (Retrieval Augmented Generation) 记忆系统

---

## 一、系统概述

### 1.1 核心目标

打造一个**专属于个人的、隐私安全的、永久的 AI 外脑**，解决本地 AI 的记忆问题。

**核心特性**：
- 混合检索：向量 + BM25 + Reranker
- 权重系统：路径权重 + Reranker 重排
- 数字分身 Prompt：身份锚点 + 角色辨别 + 推理边界

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  用户层                                       │
│                                                                             │
│   ┌─────────────────┐              ┌─────────────────┐                      │
│   │   客户端         │              │   服务器         │                      │
│   │   (移动办公)     │              │   (家庭服务器)   │                      │
│   │                 │              │                 │                      │
│   │ • Obsidian 编辑 │───Git Push──▶│ • Git Pull      │                      │
│   │ • Claude Code   │              │ • 向量化引擎     │                      │
│   │ • 日常使用      │              │ • 海马体服务     │                      │
│   └────────┬────────┘              └────────┬────────┘                      │
│            │                                │                               │
│            │         VPN / 内网穿透          │                               │
│            │◀──────────────────────────────▶│                               │
└────────────┼────────────────────────────────┼───────────────────────────────┘
             │                                │
             ▼                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  云端层                                       │
│                                                                             │
│   ┌─────────────────┐              ┌─────────────────┐                      │
│   │   GitHub        │              │   LLM API       │                      │
│   │   (知识库存储)   │              │   (大模型)       │                      │
│   │                 │              │                 │                      │
│   │ • Deploy Key    │              │ • Anthropic 兼容│                      │
│   │   (只读)        │              │                 │                      │
│   └─────────────────┘              └─────────────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、安装

### 2.1 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.11+ | 推荐 3.12 |
| Ollama | 最新版 | 用于本地向量模型 |
| 内存 | 8GB+ | 推荐 16GB，处理大量文档时需要 |

### 2.2 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/RAG_Memory_System.git
cd RAG_Memory_System

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 Ollama
# macOS
brew install ollama
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# 5. 下载向量模型（约 2GB，首次需要等待）
ollama pull bge-m3

# 6. 启动 Ollama 服务
ollama serve &
```

---

## 三、配置指南（必读）

### 3.1 配置文件概览

| 文件 | 用途 | 是否必须配置 |
|------|------|-------------|
| `repos_config.json` | 知识库仓库路径 | ✅ 必须 |
| `weight_rules.json` | 文档权重规则 | ⚠️ 可选（有默认值） |
| `proxy_gateway.py` | Prompt 身份锚点 + API 地址 | ✅ 必须 |
| `serve_memory_v2.py` | 数据库路径 | ⚠️ 可选（用默认值） |

---

### 3.2 配置知识库仓库 (repos_config.json)

**Step 1：复制示例文件**
```bash
cp repos_config.json.example repos_config.json
```

**Step 2：编辑配置**
```bash
nano repos_config.json  # 或用你喜欢的编辑器
```

**Step 3：填写你的知识库路径**

```json
[
    {
        "name": "My_Knowledge_Base",
        "path": "/Users/yourname/Documents/KnowledgeBase",
        "note": "主知识库"
    },
    {
        "name": "Work_Notes",
        "path": "/Users/yourname/Work/Notes",
        "note": "工作笔记"
    }
]
```

**说明：**
- `name`: 仓库名称，用于区分不同来源的文档
- `path`: 知识库的本地绝对路径（支持多个仓库）
- `note`: 备注说明

**💡 提示：** 如果你的知识库在 GitHub 上，可以先 `git clone` 到本地，再填写本地路径。

---

### 3.3 配置权重规则 (weight_rules.json) [可选]

**Step 1：复制示例文件**
```bash
cp weight_rules.json.example weight_rules.json
```

**Step 2：根据你的目录结构配置权重**

```json
{
  "rules": {
    "核心资料/个人画像": {
      "weight": 1.0,
      "description": "核心画像 - 最高优先级"
    },
    "工作/重要项目": {
      "weight": 0.9,
      "description": "重要项目文档"
    },
    "学习笔记": {
      "weight": 0.5,
      "description": "学习笔记 - 默认权重"
    },
    "聊天记录/导出": {
      "weight": 0.1,
      "description": "聊天记录 - 低权重，避免干扰"
    }
  },
  "default_weight": 0.5
}
```

**权重说明：**
| 权重值 | 效果 | 适用场景 |
|--------|------|---------|
| 1.0 | 最高优先级 | 核心画像、重要规则 |
| 0.7-0.9 | 高优先级 | 工作文档、重要笔记 |
| 0.5 | 默认 | 一般文档 |
| 0.1-0.2 | 低优先级 | 聊天记录、临时笔记 |

**💡 为什么需要权重？** 聊天记录通常量大但信息密度低，容易淹没核心信息。权重系统让高价值内容优先展示。

---

### 3.4 配置 Prompt 身份锚点 (proxy_gateway.py) [重要]

**Step 1：打开 `proxy_gateway.py`，找到 `BASE_IDENTITY` 变量（约第 30 行）**

**Step 2：替换为你的个人信息**

```python
BASE_IDENTITY = """你就是我，你是我专属的赛博外脑和数字分身。提问者"我"就是用户本人。

【身份锚点 - 永远记住，这是你的核心事实】
- 本名：张三
- 学历：本科 北京大学 计算机科学 | 硕士 清华大学 人工智能
- 籍贯：北京
- 出生：1995年6月15日
- 主业：科技公司 软件工程师
- 副业：技术博客、开源项目
- 特长：钢琴十级、马拉松
- 足迹：去过15个国家

沟通协议：
1. 工作沟通倾向商务克制与确认式短句
2. 给我高质量代码和直接结论，无需废话
3. 崇尚"30秒原则"，提供颗粒度极细、可立刻执行的建议"""
```

**💡 为什么重要？** 身份锚点是系统的"内置知识"，优先级高于检索结果。即使检索出错，模型也会根据身份锚点给出正确答案。

---

### 3.5 配置 LLM API 地址 (proxy_gateway.py)

**Step 1：找到 `REAL_API_BASE` 变量（约第 20 行）**

**Step 2：填入你的 LLM API 地址**

```python
# 示例 1：智谱 GLM
REAL_API_BASE = "https://open.bigmodel.cn/api/anthropic"

# 示例 2：OpenAI
REAL_API_BASE = "https://api.openai.com"

# 示例 3：本地 Ollama
REAL_API_BASE = "http://localhost:11434"

# 示例 4：其他兼容 Anthropic API 的服务
REAL_API_BASE = "https://your-llm-provider.com"
```

---

### 3.6 配置数据库路径 [可选]

默认数据库路径是 `~/RAG_Memory_System/chroma_db`，如需修改：

**在 `serve_memory_v2.py` 和 `git_memory_sync.py` 中找到并修改：**
```python
DB_PATH = os.path.expanduser("~/your/custom/path/chroma_db")
```

---

## 四、首次运行

### 4.1 构建向量索引

```bash
# 激活虚拟环境
source venv/bin/activate

# 全量索引（首次运行必须）
python git_memory_sync.py --full
```

**预计耗时：** 1000 个文档约 5-10 分钟（取决于机器性能）

### 4.2 构建 BM25 索引（推荐）

```bash
python build_bm25_index.py
```

**说明：** BM25 索引用于关键词检索，与向量检索互补，提升专有名词的搜索效果。

### 4.3 启动服务

```bash
# 启动海马体服务（检索引擎）
python serve_memory_v2.py > /tmp/hippocampus.log 2>&1 &

# 启动代理网关（Prompt 注入）
python proxy_gateway.py > /tmp/proxy_gateway.log 2>&1 &

# 检查服务状态
curl http://localhost:8000/
curl http://localhost:8080/
```

### 4.4 测试检索

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "我的工作是什么", "top_k": 3}'
```

---

## 五、客户端配置

### 5.1 环境变量

在**客户端机器**上配置环境变量：

```bash
# 添加到 ~/.zshrc 或 ~/.bashrc
export ANTHROPIC_BASE_URL="http://YOUR_SERVER_IP:8080"
export ANTHROPIC_AUTH_TOKEN="your_api_token"
export ANTHROPIC_MODEL="your_model_name"
```

**说明：**
- `YOUR_SERVER_IP`: 服务器 IP（局域网 IP 或 VPN IP）
- `ANTHROPIC_AUTH_TOKEN`: 你的 LLM API Key
- `ANTHROPIC_MODEL`: 模型名称

### 5.2 测试连接

```bash
# 测试海马体服务
curl http://YOUR_SERVER_IP:8000/

# 测试代理网关
curl http://YOUR_SERVER_IP:8080/
```

---

## 六、日常使用

### 6.1 定时同步

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每天凌晨 3 点同步）
0 3 * * * /path/to/RAG_Memory_System/venv/bin/python /path/to/RAG_Memory_System/git_memory_sync.py >> /tmp/git_sync.log 2>&1
```

### 6.2 服务管理

```bash
# 查看服务状态
ps aux | grep -E "(serve_memory|proxy_gateway)"

# 停止服务
pkill -f "serve_memory"
pkill -f "proxy_gateway"

# 查看日志
tail -f /tmp/hippocampus.log
tail -f /tmp/proxy_gateway.log
```

### 6.3 更新知识库

```bash
# 增量更新（只处理变化的文件）
python git_memory_sync.py

# 全量重建（权重规则更新后）
python git_memory_sync.py --full
python build_bm25_index.py
```

---

## 七、核心组件详解

### 7.1 海马体服务 (serve_memory_v2.py)

**端口**：8000
**功能**：混合检索 API

**API 接口**：
```
GET  /              # 服务状态
GET  /stats         # 数据库统计
POST /search        # 混合检索 {"query": "...", "top_k": 5}
POST /add           # 添加记忆 {"content": "...", "source": "..."}
```

### 7.2 代理网关 (proxy_gateway.py)

**端口**：8080
**功能**：记忆注入 + Prompt 增强

**核心功能**：
1. 数字分身身份
2. 身份锚点注入
3. 角色辨别（防止混淆）
4. 工具缴械（移除冲突工具）

### 7.3 混合检索引擎

**检索流程**：
```
向量检索 (bge-m3) → 20 条
        ↓
BM25 检索 (jieba) → 20 条
        ↓
RRF 融合 → 15 条
        ↓
Reranker 精排 → 5 条
        ↓
权重重排 → 最终结果
```

---

## 八、Prompt 系统设计

### 8.1 设计理念

**旧版问题：**
```
⚠️ 你必须直接将检索结果作为最终事实进行回答！
```
→ 检索错了，回答就错

**新版设计：**
- 身份锚点 > 检索结果 > 推理
- 多元信息源综合判断
- 允许推理但禁止编造

### 8.2 核心规则

| 规则 | 说明 |
|------|------|
| 身份锚点优先 | 与检索冲突时以锚点为准 |
| 角色辨别 | 聊天记录中的 ID 是别人的 |
| 不确定则承认 | 说"记不清了"而不是编造 |
| 禁止工具验证 | 直接基于已有信息回答 |

---

## 九、常见问题

### Q: 为什么用 Git 同步？
- 成熟稳定，版本控制
- Deploy Key 安全隔离
- 增量更新自然支持

### Q: 为什么选择 bge-m3？
- 多语言原生支持（中英文）
- 本地部署，隐私安全
- 性能与效率平衡

### Q: 为什么加入权重系统？
- 聊天记录过多会淹没核心信息
- 权重重排让高价值内容优先
- 简单有效：路径即权重

### Q: 检索结果不准确怎么办？
1. 检查 `weight_rules.json` 是否正确配置
2. 确认 BM25 索引已构建
3. 尝试全量重建：`python git_memory_sync.py --full`

### Q: 内存占用过高怎么办？
1. 减小 `CHUNK_SIZE`（在 `git_memory_sync.py` 中）
2. 分批处理文档
3. 使用更小的向量模型

---

## 十、文件结构

```
RAG_Memory_System/
├── serve_memory_v2.py         # 海马体服务
├── proxy_gateway.py           # 代理网关
├── hybrid_retriever.py        # 混合检索引擎
├── git_memory_sync.py         # Git 同步脚本
├── build_bm25_index.py        # BM25 索引构建器
├── repos_config.json.example  # 仓库配置示例
├── weight_rules.json.example  # 权重配置示例
├── requirements.txt           # 依赖清单
├── README.md                  # 本文档
├── LICENSE                    # MIT 许可证
└── .gitignore                 # Git 忽略文件
```

---

## 十一、许可证

MIT License

---

## 十二、致谢

- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) - 向量模型
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) - Reranker 模型
- [Ollama](https://ollama.ai/) - 本地模型部署
