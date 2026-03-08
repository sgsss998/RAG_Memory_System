# RAG Memory System - 私有化 AI 记忆系统

> 一个专属于个人的、隐私安全的 RAG (Retrieval Augmented Generation) 记忆系统

---

## 一、系统架构总览

### 1.1 核心目标

打造一个**专属于个人的、隐私安全的、永久的 AI 外脑**，解决本地 AI 的记忆问题。

**核心特性**：
- 混合检索：向量 + BM25 + Reranker
- 权重系统：路径权重 + Reranker 重排
- 数字分身 Prompt：身份锚点 + 角色辨别 + 推理边界

### 1.2 六层架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户交互层                                       │
│  Claude Code (MacBook) ──→ ANTHROPIC_BASE_URL=http://SERVER_IP:8080         │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Layer 1: 代理网关 (proxy_gateway.py)                      │
│  ├─ 端口: 8080                                                               │
│  ├─ 框架: FastAPI + Uvicorn                                                  │
│  ├─ 功能: 请求拦截 → 用户问题提取 → 记忆注入 → 请求转发                        │
│  ├─ 特性: 数字分身身份 + 身份锚点注入 + 角色辨别 + 工具缴械                     │
│  └─ 耦合: 与海马体服务 HTTP 通信，与 LLM API HTTP 通信                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                            ┌───────────┴───────────┐
                            ▼                       ▼
┌──────────────────────────────────┐   ┌──────────────────────────────────────┐
│  Layer 2: 海马体检索服务           │   │  Layer 3: LLM API                     │
│  (serve_memory_v2.py)            │   │  (Anthropic 兼容 API)                 │
│  ├─ 端口: 8000                    │   │  ├─ 协议: Anthropic API 兼容          │
│  ├─ 框架: FastAPI                 │   │  ├─ 支持: Claude / GLM / OpenAI 等    │
│  ├─ 功能: 混合检索调度             │   │  └─ 限制: 仅支持 user/assistant role  │
│  └─ 耦合: 调用 HybridRetriever    │   └──────────────────────────────────────┘
└──────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Layer 4: 混合检索引擎 (hybrid_retriever.py)                │
│  ├─ 向量检索: ChromaDB + bge-m3                                              │
│  ├─ 关键词检索: BM25Okapi + jieba                                            │
│  ├─ 融合算法: RRF (Reciprocal Rank Fusion)                                   │
│  ├─ 精排模型: bge-reranker-v2-m3                                             │
│  └─ 权重重排: 路径权重 + Reranker 分数加权                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────────────────┐   ┌──────────────────────────────────────────────┐
│  Layer 5a: 向量数据库     │   │  Layer 5b: BM25 索引                          │
│  (ChromaDB)              │   │  (bm25_index.pkl)                            │
│  ├─ 路径: ~/RAG_System/  │   │  ├─ 分词器: jieba (中文)                      │
│  ├─ 集合: memory_v1      │   │  ├─ 算法: BM25Okapi                          │
│  ├─ 索引: HNSW           │   │  └─ 存储: pickle 序列化                       │
│  └─ 距离: L2             │   └──────────────────────────────────────────────┘
└──────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Layer 6: 向量化引擎 (Ollama + bge-m3)                      │
│  ├─ 模型: BAAI/bge-m3                                                        │
│  ├─ 参数量: 566M                                                             │
│  ├─ 向量维度: 1024                                                           │
│  ├─ 上下文: 8192 tokens                                                      │
│  └─ 特点: 多语言原生支持，中英文效果优秀                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                    ▲
                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Layer 7: 数据摄入管道 (git_memory_sync.py)                 │
│  ├─ 数据源: Markdown 文件 (Obsidian Vault)                                   │
│  ├─ 切片策略: 双重语义切片                                                    │
│  ├─ 去重机制: MD5 缓存                                                        │
│  ├─ 权重计算: 路径匹配                                                        │
│  └─ 输出: 向量化后存入 ChromaDB                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、Pipeline 详细分解

### 2.1 阶段一：数据摄入 (git_memory_sync.py)

#### 2.1.1 数据源

```python
# 支持多仓库配置
repos = [
    {
        "name": "Knowledge_Base",
        "path": "/Users/xxx/Documents/KnowledgeBase",
        "note": "主知识库"
    }
]

# 支持的文件格式
SUPPORTED_FORMATS = [".md"]  # 目前仅支持 Markdown
```

#### 2.1.2 Git 同步机制

```python
def run_git_pull(repo_path: str) -> bool:
    """执行 git pull，拉取最新知识库"""
    result = subprocess.run(
        ['git', 'pull'],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=60
    )
    return result.returncode == 0
```

#### 2.1.3 MD5 去重机制

```python
MD5_CACHE = ".md5_cache.json"  # 缓存文件路径

def get_file_md5(filepath: str) -> str:
    """计算文件 MD5，用于增量更新"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# 增量更新逻辑
current_md5 = get_file_md5(filepath)
cached_md5 = md5_cache.get(filepath)

if cached_md5 == current_md5:
    continue  # 跳过未变化的文件

md5_cache[filepath] = current_md5  # 更新缓存
```

**去重策略：**
- 文件级：MD5 校验，只处理变化的文件
- 切片级：删除旧向量 → 重新索引

---

### 2.2 阶段二：双重语义切片

#### 2.2.1 第一重：MarkdownHeaderTextSplitter (物理隔离)

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),    # 一级标题
    ("##", "Header 2"),   # 二级标题
    ("###", "Header 3"),  # 三级标题
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # 保留标题作为上下文锚点
)
```

**算法原理：**
- 基于 Markdown 语法树的硬边界切分
- 每个 `#` `##` `###` 标题自动成为切片边界
- 保留标题文本在切片中，维持语义完整性

**切分示例：**
```
原始文档:
# 第一章 个人背景
内容A

## 1.1 教育经历
内容B

## 1.2 工作经历
内容C

切分后:
Chunk 1: "# 第一章 个人背景\n内容A"
Chunk 2: "# 第一章 个人背景\n## 1.1 教育经历\n内容B"
Chunk 3: "# 第一章 个人背景\n## 1.2 工作经历\n内容C"
```

#### 2.2.2 第二重：RecursiveCharacterTextSplitter (柔性约束)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,           # 目标长度 (字符)
    chunk_overlap=50,         # 重叠区 (防止语义断裂)
    separators=[              # 分隔符优先级
        "\n\n",               # 1. 段落边界 (最高优先级)
        "\n",                 # 2. 换行
        "。",                 # 3. 中文句号
        "，",                 # 4. 中文逗号
        " ",                  # 5. 空格
        ""                    # 6. 强制切分 (最后手段)
    ]
)
```

**递归切分流程：**
```
尝试用 "\n\n" 切
    ↓ 如果某块仍 > 300
尝试用 "\n" 切
    ↓ 如果某块仍 > 300
尝试用 "。" 切
    ↓ 如果某块仍 > 300
尝试用 "，" 切
    ↓ 如果某块仍 > 300
强制按字符切 (最后手段)
```

**重叠机制：**
- 保留 50 字符重叠
- 防止关键信息在边界丢失
- 例如：`...这是前一段的结尾。这是后一段的开头...`

#### 2.2.3 过滤策略

```python
# 文件级过滤
if len(content.strip()) < 50:
    continue  # 跳过空白或过短文件

# 切片级过滤
chunks = [c for c in chunks if len(c.strip()) > 20]  # 过滤过短切片
```

---

### 2.3 阶段三：权重计算

#### 2.3.1 权重规则配置

```json
{
  "rules": {
    "核心文档/人物画像": {
      "weight": 1.0,
      "description": "核心画像 - 最高优先级"
    },
    "工作文档/重要项目": {
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

#### 2.3.2 权重计算算法

```python
def calculate_weight(rel_path: str, weight_rules: dict) -> float:
    """根据文件路径计算权重"""
    rules = weight_rules.get("rules", {})
    default = weight_rules.get("default_weight", 0.5)

    # 路径匹配：第一个匹配的规则生效
    for pattern, rule in rules.items():
        if pattern in rel_path:
            return rule.get("weight", default)

    return default
```

**权重值含义：**
| 权重值 | 效果 | 适用场景 |
|--------|------|---------|
| 1.0 | 最高优先级 | 核心画像、重要规则 |
| 0.7-0.9 | 高优先级 | 工作文档、重要笔记 |
| 0.5 | 默认 | 一般文档 |
| 0.1-0.2 | 低优先级 | 聊天记录、临时笔记 |

---

### 2.4 阶段四：向量化

#### 2.4.1 模型配置

| 属性 | 值 |
|------|-----|
| 模型名 | BAAI/bge-m3 |
| 部署方式 | Ollama 本地推理 |
| 参数量 | 566M |
| 向量维度 | 1024 |
| 最大上下文 | 8192 tokens |
| 推理框架 | llama.cpp (量化) |
| 量化等级 | Q4_K_M (4-bit) |
| 模型大小 | ~2GB |

#### 2.4.2 向量化调用

```python
import ollama

# 输入: 纯文本切片
response = ollama.embeddings(
    model="bge-m3",
    prompt=chunk  # 待向量化的文本
)

# 输出: 1024 维浮点向量
embedding = response["embedding"]  # List[float], len=1024
```

#### 2.4.3 模型特点

- **多语言支持**：原生支持中英文混合，无需翻译
- **语义理解**：基于 BERT 架构，深层上下文编码
- **长文本支持**：8192 tokens 上下文，远超传统 512
- **零样本检索**：无需微调即可使用
- **隐私安全**：本地部署，数据不出境

---

### 2.5 阶段五：向量存储 (ChromaDB)

#### 2.5.1 数据库初始化

```python
import chromadb

DB_PATH = os.path.expanduser("~/RAG_Memory_System/chroma_db")

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="memory_v1_semantic",
    metadata={"description": "私有知识库 - 语义切片版"}
)
```

#### 2.5.2 存储结构

```python
collection.upsert(
    ids=[chunk_id],                  # 唯一标识: "{repo_name}:{rel_path}:{index}"
    embeddings=[embedding],          # 1024 维向量
    documents=[chunk],               # 原始文本
    metadatas=[{                     # 元数据
        "source": f"{repo_name}/{rel_path}",
        "weight": weight,            # 路径权重
        "chunk_index": idx,
        "total_chunks": len(chunks)
    }]
)
```

#### 2.5.3 底层实现

| 组件 | 技术 |
|------|------|
| 索引算法 | HNSW (Hierarchical Navigable Small World) |
| 距离度量 | L2 (欧几里得距离) |
| 持久化 | SQLite + Arrow 格式 |
| 并发 | 单进程，支持多客户端读取 |

#### 2.5.4 HNSW 算法原理

```
图结构示意:
Layer 2:     A ─────── B
             │         │
Layer 1:     C───D───E─F
             │   │   │ │
Layer 0:     G─H─I─J─K─L─M (所有向量节点)

查询过程:
1. 从 Layer 2 的入口点 A 开始
2. 在当前层贪婪搜索最近邻
3. 逐层向下细化
4. 最终在 Layer 0 找到 Top-K

复杂度: O(log N)
```

**HNSW 参数 (ChromaDB 默认)：**
- `hnsw:space`: "l2"
- `hnsw:construction_ef`: 100 (构建时搜索范围)
- `hnsw:M`: 16 (每层最大连接数)

---

## 三、混合检索引擎详解

### 3.1 检索流程图

```
用户查询: "我的工作是什么？"
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Step 1: 查询向量化                                        │
│  ollama.embeddings(model="bge-m3", prompt=query)          │
│  → 1024 维查询向量                                         │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Step 2: 并行双路检索                                      │
│  ┌─────────────────┐    ┌─────────────────┐               │
│  │ 向量检索 (ChromaDB)│    │ BM25 检索 (jieba) │              │
│  │ → 20 条候选       │    │ → 20 条候选       │               │
│  └─────────────────┘    └─────────────────┘               │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Step 3: RRF 融合 (Reciprocal Rank Fusion)                │
│  score = Σ(1 / (k + rank))  where k=60                    │
│  → 15 条融合后候选                                         │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Step 4: Reranker 精排 (bge-reranker-v2-m3)               │
│  CrossEncoder.predict([(query, doc1), (query, doc2), ...]) │
│  → 计算每对 (query, doc) 的相关性分数                       │
│  → 5 条精排结果                                            │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Step 5: 权重重排                                          │
│  final_score = rerank_score × (1 + weight)                │
│  → 最终 Top-K 结果                                         │
└───────────────────────────────────────────────────────────┘
```

### 3.2 向量检索 (hybrid_retriever.py)

```python
def vector_search(self, query: str, top_k: int = 20) -> List[Dict]:
    """向量检索"""
    # 1. 查询向量化
    response = ollama.embeddings(model="bge-m3", prompt=query)
    query_embedding = response["embedding"]

    # 2. ChromaDB 检索
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 3. 格式化结果
    formatted = []
    for i in range(len(results['ids'][0])):
        formatted.append({
            'id': results['ids'][0][i],
            'document': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i],
            'score': 1 / (1 + results['distances'][0][i]),  # 距离转相似度
            'source': 'vector'
        })

    return formatted
```

**L2 距离公式：**
```
distance = sqrt(Σ(qi - di)²)

其中:
- q: 查询向量 (1024 维)
- d: 文档向量 (1024 维)
- distance 越小，相似度越高
```

### 3.3 BM25 检索 (build_bm25_index.py)

#### 3.3.1 索引构建

```python
from rank_bm25 import BM25Okapi
import jieba

# 1. 读取所有文档
documents = collection.get(include=["documents", "metadatas"])

# 2. 中文分词
tokenized_docs = []
for doc in documents['documents']:
    tokens = list(jieba.cut(doc))  # jieba 精确模式分词
    tokenized_docs.append(tokens)

# 3. 构建 BM25 索引
bm25 = BM25Okapi(tokenized_docs)

# 4. 持久化
index_data = {
    'bm25': bm25,
    'documents': documents['documents'],
    'metadatas': documents['metadatas'],
    'ids': documents['ids']
}
pickle.dump(index_data, open("bm25_index.pkl", 'wb'))
```

#### 3.3.2 BM25 检索

```python
def bm25_search(self, query: str, top_k: int = 20) -> List[Dict]:
    """BM25 检索"""
    # 1. 中文分词
    tokenized_query = list(jieba.cut(query))

    # 2. 计算 BM25 分数
    scores = self.bm25.get_scores(tokenized_query)

    # 3. 获取 Top-K
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    # 4. 格式化结果
    return [{
        'id': self.bm25_ids[idx],
        'document': self.bm25_documents[idx],
        'metadata': self.bm25_metadatas[idx],
        'score': scores[idx],
        'source': 'bm25'
    } for idx in top_indices]
```

**BM25 公式：**
```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))

其中:
- f(qi, D): 词 qi 在文档 D 中的词频
- |D|: 文档 D 的长度
- avgdl: 平均文档长度
- k1, b: 调节参数 (通常 k1=1.5, b=0.75)
```

**为什么需要 BM25？**
- 向量检索对专有名词（人名、地名、ID）效果差
- BM25 精确匹配关键词，弥补向量检索的不足

### 3.4 RRF 融合 (Reciprocal Rank Fusion)

```python
def reciprocal_rank_fusion(
    self,
    results_list: List[List[Dict]],
    k: int = 60,
    max_candidates: int = 15
) -> List[Dict]:
    """
    RRF 算法: 融合多路检索结果
    公式: score(doc) = Σ 1/(k + rank(doc))
    """
    doc_scores = {}

    for results in results_list:
        for rank, item in enumerate(results):
            doc_key = item['document']  # 用文档内容作为 key

            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    'id': item['id'],
                    'document': item['document'],
                    'metadata': item['metadata'],
                    'rrf_score': 0,
                    'sources': []
                }

            # RRF 公式
            doc_scores[doc_key]['rrf_score'] += 1 / (k + rank + 1)
            doc_scores[doc_key]['sources'].append(item['source'])

    # 按 RRF 分数排序，限制候选数量
    sorted_docs = sorted(
        doc_scores.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )

    return sorted_docs[:max_candidates]
```

**RRF 示例：**
```
文档 A: 向量检索 rank=1, BM25 检索 rank=3
RRF(A) = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

文档 B: 向量检索 rank=5, BM25 检索 rank=1
RRF(B) = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318

→ 文档 A 排名更高 (因为两路检索都靠前)
```

### 3.5 Reranker 精排 (bge-reranker-v2-m3)

```python
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self):
        # 加载 Reranker 模型
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            max_length=512,
            cache_folder="~/RAG_Memory_System/models",
            device=self.device  # mps (Apple Silicon) / cuda / cpu
        )

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """Reranker 精排 + 权重加权"""
        # 1. 构造 query-doc 对
        pairs = [(query, item['document']) for item in candidates]

        # 2. 计算相关性分数
        scores = self.reranker.predict(pairs)

        # 3. 按分数重新排序（加入权重）
        reranked = []
        for i, item in enumerate(candidates):
            rerank_score = float(scores[i])
            weight = item.get('metadata', {}).get('weight', 0.5)

            # 最终分数 = rerank分数 × 权重放大因子
            weight_boost = 1 + weight
            final_score = rerank_score * weight_boost

            item['rerank_score'] = rerank_score
            item['weight'] = weight
            item['final_score'] = final_score
            reranked.append(item)

        # 4. 按最终分数排序
        reranked.sort(key=lambda x: x['final_score'], reverse=True)

        return reranked[:top_k]
```

**Reranker vs 向量模型：**
| 对比项 | 向量模型 (bge-m3) | Reranker (bge-reranker) |
|--------|-------------------|-------------------------|
| 输入 | 单文本 | (query, doc) 对 |
| 输出 | 向量 | 相关性分数 |
| 精度 | 较低 | 较高 |
| 速度 | 快 (预先计算) | 慢 (实时计算) |
| 用途 | 召回 | 精排 |

---

## 四、Prompt 系统设计

### 4.1 代理网关架构 (proxy_gateway.py)

```python
# 请求处理流程
用户请求 → 代理网关 (8080)
    │
    ├─→ 1. 提取用户问题
    │
    ├─→ 2. 调用海马体检索 (8000)
    │
    ├─→ 3. 构建注入内容
    │       ├─ 身份锚点 (BASE_IDENTITY)
    │       ├─ 检索结果 (memory_context)
    │       └─ 回答规则 (answering_rules)
    │
    ├─→ 4. 注入到用户消息开头
    │
    ├─→ 5. 工具缴械 (移除冲突工具)
    │
    └─→ 6. 转发到 LLM API
```

### 4.2 身份锚点设计

```python
BASE_IDENTITY = """你就是我，你是我专属的赛博外脑和数字分身。提问者"我"就是用户本人。

【身份锚点 - 永远记住，这是你的核心事实】
- 本名：[你的名字]
- 学历：[你的学历信息]
- 籍贯：[你的籍贯]
- 主业：[你的工作]
- 副业：[你的副业]
- 特长：[你的特长]
- 足迹：[你的经历]

沟通协议：
1. 工作沟通倾向商务克制与确认式短句
2. 给我高质量代码和直接结论，无需废话
3. 崇尚"30秒原则"，提供颗粒度极细、可立刻执行的建议"""
```

### 4.3 记忆注入策略

```python
def inject_memory_to_request(body: dict, user_question: str, memory_context: str) -> dict:
    """把记忆注入到第一条 user 消息的开头"""

    rag_injection = f"""<memory_slices>
{memory_context}
</memory_slices>

<answering_rules>
【回答优先级 - 从高到低】
1. 身份锚点中的固定信息 → 最高优先级，直接用
2. 记忆切片中明确是"我"说的内容 → 直接用
3. 基于以上两点的合理推理 → 可以用
4. 完全没有依据的猜测 → 禁止！说"记不清了"

【角色辨别规则 - 重要！】
- 记忆切片中的 ID、微信号、手机号通常是【聊天对象的】，不是你的
- 聊天记录文件名中的信息属于【对话参与者】，需判断是谁说的
- 当身份锚点与检索结果冲突时，以身份锚点为准
- 不确定是谁说的内容，不要当成自己的

【语言风格】
- 禁止"根据知识库"、"检索结果显示"等机械词汇
- 用第一人称回答，就像在回忆自己的事

【安全边界】
- 禁止调用任何工具验证信息，直接基于已有信息回答
</answering_rules>"""

    # 注入到用户消息开头
    full_prefix = f"{BASE_IDENTITY}\n\n{rag_injection}\n\n"

    for msg in body["messages"]:
        if msg.get("role") == "user":
            msg["content"] = full_prefix + msg["content"]
            break

    return body
```

### 4.4 工具缴械机制

```python
# 需要被移除的工具（这些工具会在本地文件系统搜索，与 RAG 冲突）
DISABLED_TOOLS = ["search_memories", "read_file", "search_files"]

def disarm_conflicting_tools(body: dict) -> dict:
    """物理缴械：移除与 RAG 冲突的工具"""
    if "tools" in body and isinstance(body["tools"], list):
        allowed_tools = []

        for tool in body["tools"]:
            tool_name = tool.get("function", {}).get("name", "")

            if tool_name not in DISABLED_TOOLS:
                allowed_tools.append(tool)
            else:
                print(f"[代理] 🔒 缴械工具: {tool_name}")

        body["tools"] = allowed_tools

    return body
```

---

## 五、性能配置参数

### 5.1 检索参数

```python
# hybrid_retriever.py
VECTOR_TOP_K = 20        # 向量检索召回数量
BM25_TOP_K = 20          # BM25 检索召回数量
RERANKER_CANDIDATES = 15  # Reranker 最大候选数 (融合后)
```

### 5.2 切片参数

```python
# git_memory_sync.py
CHUNK_SIZE = 300         # 目标切片长度 (字符)
CHUNK_OVERLAP = 50       # 切片重叠长度 (字符)
```

### 5.3 性能优化建议

| 场景 | 建议配置 |
|------|---------|
| 文档量 < 1000 | 默认配置即可 |
| 文档量 1000-5000 | VECTOR_TOP_K=30, RERANKER_CANDIDATES=20 |
| 文档量 > 5000 | 考虑分库或使用 Milvus |
| 内存 < 16GB | 减小 RERANKER_CANDIDATES 或禁用 Reranker |
| 追求速度 | 禁用 Reranker，仅用向量+BM25 |
| 追求精度 | 启用全流程，适当增加召回量 |

---

## 六、耦合度分析

### 6.1 模块依赖图

```
┌─────────────┐
│ 数据源       │ Obsidian Vault (Markdown)
└──────┬──────┘
       │ 文件系统 (解耦: 可替换为任何文本源)
       ▼
┌─────────────┐
│ 切片器       │ LangChain Text Splitters
└──────┬──────┘
       │ 纯文本 (解耦: 标准字符串)
       ▼
┌─────────────┐
│ 向量化引擎   │ Ollama + bge-m3
└──────┬──────┘
       │ HTTP API (解耦: 可替换为 OpenAI/Voyage/Jina)
       ▼
┌─────────────┐
│ 向量数据库   │ ChromaDB
└──────┬──────┘
       │ Python 库 (解耦: 可替换为 Milvus/Qdrant/Weaviate)
       ▼
┌─────────────┐
│ BM25 索引   │ rank-bm25 + jieba
└──────┬──────┘
       │ pickle 文件 (解耦: 可替换为 Elasticsearch)
       ▼
┌─────────────┐
│ 混合检索器   │ HybridRetriever
└──────┬──────┘
       │ Python 类 (解耦: 可独立使用)
       ▼
┌─────────────┐
│ 海马体服务   │ serve_memory_v2.py (FastAPI)
└──────┬──────┘
       │ HTTP API (解耦: 任何语言都可调用)
       ▼
┌─────────────┐
│ 代理网关     │ proxy_gateway.py (FastAPI)
└──────┬──────┘
       │ HTTP API (解耦: 可用 Nginx/Envoy 替代)
       ▼
┌─────────────┐
│ LLM API     │ Anthropic 兼容 API
└─────────────┘
```

### 6.2 解耦能力矩阵

| 模块 | 接口类型 | 可替换性 | 替换方案 |
|------|---------|---------|---------|
| 数据源 | 文件系统 | ⭐⭐⭐⭐⭐ | PDF, DOCX, 网页, 数据库 |
| 切片器 | Python 库 | ⭐⭐⭐⭐⭐ | LlamaIndex, SemChunk, 自定义 |
| 向量化引擎 | HTTP API | ⭐⭐⭐⭐ | OpenAI, Voyage AI, Jina AI |
| 向量数据库 | Python 库 | ⭐⭐⭐⭐ | Milvus, Qdrant, Weaviate, Pinecone |
| BM25 索引 | pickle 文件 | ⭐⭐⭐⭐ | Elasticsearch, Whoosh |
| Reranker | Python 库 | ⭐⭐⭐⭐ | Cohere Rerank, Jina Reranker |
| 海马体服务 | HTTP API | ⭐⭐⭐⭐⭐ | 任何 Web 框架 |
| 代理网关 | HTTP API | ⭐⭐⭐⭐⭐ | Nginx + Lua, Envoy |
| LLM API | Anthropic 协议 | ⭐⭐⭐ | OpenAI, Claude, 本地模型 |

### 6.3 接口规范

**海马体服务 API (端口 8000)：**
```
POST /search
Content-Type: application/json

Request:
{
    "query": "用户问题",
    "top_k": 5,
    "use_hybrid": true,
    "use_reranker": true
}

Response:
{
    "status": "success",
    "mode": "hybrid",
    "query": "用户问题",
    "count": 5,
    "memories": [
        {
            "content": "检索到的内容",
            "source": "来源文件",
            "weight": 0.9,
            "score": 0.85
        }
    ]
}
```

**代理网关 API (端口 8080)：**
```
POST /v1/messages
Content-Type: application/json
x-api-key: your_api_key

Request: (Anthropic Messages API 格式)
{
    "model": "claude-3-opus-20240229",
    "max_tokens": 4096,
    "messages": [
        {"role": "user", "content": "用户问题"}
    ]
}

Response: (Anthropic Messages API 格式，流式)
```

---

## 七、安装与配置

### 7.1 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.11+ | 推荐 3.12 |
| Ollama | 最新版 | 用于本地向量模型 |
| 内存 | 8GB+ | 推荐 16GB，处理大量文档时需要 |
| 磁盘 | 5GB+ | 向量模型 ~2GB + 数据库 |

### 7.2 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/RAG_Memory_System.git
cd RAG_Memory_System

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 Ollama
# macOS
brew install ollama
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# 5. 下载向量模型（约 2GB）
ollama pull bge-m3

# 6. 启动 Ollama 服务
ollama serve &
```

### 7.3 配置文件

```bash
# 复制示例配置
cp repos_config.json.example repos_config.json
cp weight_rules.json.example weight_rules.json

# 编辑配置
nano repos_config.json   # 配置知识库路径
nano weight_rules.json   # 配置权重规则
nano proxy_gateway.py    # 配置身份锚点和 API 地址
```

---

## 八、首次运行

### 8.1 构建索引

```bash
# 激活虚拟环境
source venv/bin/activate

# 全量索引（首次运行必须）
python git_memory_sync.py --full

# 构建 BM25 索引（推荐）
python build_bm25_index.py
```

**预计耗时：** 1000 个文档约 5-10 分钟

### 8.2 启动服务

```bash
# 启动海马体服务（检索引擎）
python serve_memory_v2.py > /tmp/hippocampus.log 2>&1 &

# 启动代理网关（Prompt 注入）
python proxy_gateway.py > /tmp/proxy_gateway.log 2>&1 &

# 检查服务状态
curl http://localhost:8000/
curl http://localhost:8080/
```

### 8.3 测试检索

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "我的工作是什么", "top_k": 3}'
```

---

## 九、客户端配置

### 9.1 环境变量

```bash
# 添加到 ~/.zshrc 或 ~/.bashrc
export ANTHROPIC_BASE_URL="http://YOUR_SERVER_IP:8080"
export ANTHROPIC_AUTH_TOKEN="your_api_token"
export ANTHROPIC_MODEL="your_model_name"
```

### 9.2 Claude Code 配置

在 Claude Code 中，会自动读取环境变量：
- `ANTHROPIC_BASE_URL` → 代理网关地址
- `ANTHROPIC_AUTH_TOKEN` → LLM API Key
- `ANTHROPIC_MODEL` → 模型名称

---

## 十、常见问题

### Q: 为什么用 Git 同步？
- 成熟稳定，版本控制
- Deploy Key 安全隔离
- 增量更新自然支持

### Q: 为什么选择 bge-m3？
- 多语言原生支持（中英文）
- 本地部署，隐私安全
- 性能与效率平衡

### Q: 为什么需要 BM25？
- 向量检索对专有名词（人名、地名、ID）效果差
- BM25 精确匹配关键词，弥补向量检索的不足

### Q: 为什么需要 Reranker？
- 向量检索是"粗筛"，精度有限
- Reranker 是"精排"，计算 query-doc 对的相关性
- 权衡：速度换精度

### Q: 为什么加入权重系统？
- 聊天记录过多会淹没核心信息
- 权重重排让高价值内容优先
- 简单有效：路径即权重

### Q: 检索结果不准确怎么办？
1. 检查 `weight_rules.json` 是否正确配置
2. 确认 BM25 索引已构建
3. 尝试全量重建：`python git_memory_sync.py --full`

### Q: 内存占用过高怎么办？
1. 减小 `RERANKER_CANDIDATES`
2. 禁用 Reranker（设 `enable_reranker=False`）
3. 分批处理文档

---

## 十一、文件结构

```
RAG_Memory_System/
├── serve_memory_v2.py         # 海马体服务 (FastAPI, 端口 8000)
├── proxy_gateway.py           # 代理网关 (FastAPI, 端口 8080)
├── hybrid_retriever.py        # 混合检索引擎 (向量+BM25+Reranker)
├── git_memory_sync.py         # 数据摄入管道 (Git同步+切片+向量化)
├── build_bm25_index.py        # BM25 索引构建器
├── repos_config.json.example  # 仓库配置示例
├── weight_rules.json.example  # 权重配置示例
├── requirements.txt           # Python 依赖
├── README.md                  # 本文档
├── LICENSE                    # MIT 许可证
└── .gitignore                 # Git 忽略文件
```

---

## 十二、致谢

- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) - 向量模型
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) - Reranker 模型
- [Ollama](https://ollama.ai/) - 本地模型部署
- [LangChain](https://python.langchain.com/) - 文本切分器
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) - BM25 实现
- [jieba](https://github.com/fxsjy/jieba) - 中文分词
