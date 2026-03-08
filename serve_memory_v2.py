#!/usr/bin/env python3
"""
海马体检索服务 v2.0 - 支持混合检索
新增功能：BM25 + 向量检索 + Reranker
运行方式: python3 serve_memory_v2.py
服务地址: http://0.0.0.0:8000
"""

import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import uvicorn
from datetime import datetime
from typing import Optional

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置 - 根据你的环境修改
DB_PATH = os.path.expanduser("~/RAG_Memory_System/chroma_db")
HOST = "0.0.0.0"  # 允许局域网访问
PORT = 8000
ENABLE_HYBRID = os.path.exists(os.path.expanduser("~/RAG_Memory_System/bm25_index.pkl"))

app = FastAPI(title="🧠 海马体记忆服务 v2.0", version="2.0")

# 全局变量：混合检索器
hybrid_retriever = None

# 初始化数据库连接
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="memory_v1_semantic",
    metadata={"description": "私有知识库 - 语义切片版"}
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_hybrid: bool = True  # 是否使用混合检索
    use_reranker: bool = True  # 是否使用 Reranker

class AddRequest(BaseModel):
    content: str
    source: str = "manual"

@app.on_event("startup")
async def startup_event():
    """启动时初始化混合检索器"""
    global hybrid_retriever, ENABLE_HYBRID

    if ENABLE_HYBRID:
        try:
            print("🔄 初始化混合检索器...")
            from hybrid_retriever import HybridRetriever
            hybrid_retriever = HybridRetriever(
                enable_reranker=True,
                enable_bm25=True
            )
            print("✅ 混合检索器初始化成功")
        except Exception as e:
            print(f"⚠️ 混合检索器初始化失败: {e}")
            print("   将回退到纯向量检索模式")
            ENABLE_HYBRID = False
    else:
        print("ℹ️ BM25 索引不存在，使用纯向量检索模式")

@app.get("/")
def root():
    """服务状态"""
    return {
        "status": "online",
        "service": "海马体记忆检索 v2.0",
        "mode": "hybrid" if ENABLE_HYBRID else "vector_only",
        "time": datetime.now().isoformat(),
        "endpoints": ["/search", "/add", "/stats"]
    }

@app.post("/search")
def search_memory(req: QueryRequest):
    """检索相关记忆（支持混合检索）"""
    try:
        # 如果启用混合检索且混合检索器可用
        if req.use_hybrid and ENABLE_HYBRID and hybrid_retriever:
            print(f"[混合检索] 查询: {req.query}")
            results = hybrid_retriever.search(
                req.query,
                top_k=req.top_k,
                use_reranker=req.use_reranker
            )

            # 格式化结果
            memories = []
            for item in results:
                memories.append({
                    "content": item['document'],
                    "source": item['metadata'].get("source", "unknown"),
                    "weight": item.get('weight', item.get('metadata', {}).get('weight', 0.5)),
                    "score": item.get('final_score', item.get('rerank_score', item.get('rrf_score', 0))),
                    "sources": item.get('sources', [])
                })

            return {
                "status": "success",
                "mode": "hybrid",
                "query": req.query,
                "count": len(memories),
                "memories": memories
            }

        # 回退到纯向量检索
        else:
            print(f"[向量检索] 查询: {req.query}")
            import ollama

            # 向量化查询
            response = ollama.embeddings(model="bge-m3", prompt=req.query)
            query_embedding = response["embedding"]

            # 搜索最相似的记录
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=req.top_k,
                include=["documents", "metadatas", "distances"]
            )

            # 格式化结果
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    memories.append({
                        "content": doc,
                        "source": results['metadatas'][0][i].get("source", "unknown") if results['metadatas'] else "unknown",
                        "distance": results['distances'][0][i] if results['distances'] else 0
                    })

            return {
                "status": "success",
                "mode": "vector_only",
                "query": req.query,
                "count": len(memories),
                "memories": memories
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/add")
def add_memory(req: AddRequest):
    """手动添加记忆"""
    try:
        import ollama

        # 向量化
        response = ollama.embeddings(model="bge-m3", prompt=req.content)
        embedding = response["embedding"]

        # 生成 ID
        doc_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 存储
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[req.content],
            metadatas=[{"source": req.source, "type": "manual"}]
        )

        return {"status": "success", "id": doc_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/stats")
def get_stats():
    """获取数据库统计"""
    try:
        count = collection.count()
        return {
            "status": "success",
            "total_memories": count,
            "db_path": DB_PATH,
            "hybrid_mode": ENABLE_HYBRID,
            "bm25_index": os.path.exists(os.path.expanduser("~/RAG_Memory_System/bm25_index.pkl"))
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("=" * 60)
    print(f"🧠 海马体服务启动中... v2.0")
    print("=" * 60)
    print(f"📡 监听地址: http://{HOST}:{PORT}")
    print(f"💾 数据库路径: {DB_PATH}")
    print(f"🔀 混合检索: {'已启用' if ENABLE_HYBRID else '未启用（BM25 索引不存在）'}")
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    uvicorn.run(app, host=HOST, port=PORT)
