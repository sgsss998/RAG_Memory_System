#!/usr/bin/env python3
"""
🔀 混合检索引擎 (性能优化版)
结合向量检索 + BM25 + Reranker
优化: GPU 加速 + 候选集控制 + 计时日志
"""

import os
import pickle
import jieba
import time
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List, Dict
import chromadb
import torch

# ============== 配置 ==============
DB_PATH = os.path.expanduser("~/RAG_Memory_System/chroma_db")
BM25_INDEX_PATH = os.path.expanduser("~/RAG_Memory_System/bm25_index.pkl")
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_CACHE = os.path.expanduser("~/RAG_Memory_System/models")

# 性能配置
VECTOR_TOP_K = 20      # 向量检索数量
BM25_TOP_K = 20        # BM25检索数量
RERANKER_CANDIDATES = 15  # Reranker 最大候选数

class HybridRetriever:
    """混合检索器：向量 + BM25 + Reranker (性能优化版)"""

    def __init__(self, enable_reranker=True, enable_bm25=True):
        self.enable_reranker = enable_reranker
        self.enable_bm25 = enable_bm25

        # 检测可用设备
        self.device = self._detect_device()
        print(f"🖥️ 计算设备: {self.device}")

        # 1. 加载 ChromaDB
        print("📂 加载向量数据库...")
        client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = client.get_collection("memory_v1_semantic")
        print(f"✅ 向量数据库加载完成: {self.collection.count():,} 条记录")

        # 2. 加载 BM25 索引
        if self.enable_bm25:
            if os.path.exists(BM25_INDEX_PATH):
                print("📂 加载 BM25 索引...")
                with open(BM25_INDEX_PATH, 'rb') as f:
                    index_data = pickle.load(f)
                self.bm25 = index_data['bm25']
                self.bm25_documents = index_data['documents']
                self.bm25_metadatas = index_data['metadatas']
                self.bm25_ids = index_data['ids']
                print(f"✅ BM25 索引加载完成: {len(self.bm25_documents):,} 条文档")
            else:
                print(f"⚠️ BM25 索引不存在: {BM25_INDEX_PATH}")
                self.enable_bm25 = False

        # 3. 加载 Reranker
        if self.enable_reranker:
            try:
                print("🔧 加载 Reranker 模型...")
                start_time = time.time()

                self.reranker = CrossEncoder(
                    RERANKER_MODEL,
                    max_length=512,
                    cache_folder=RERANKER_CACHE,
                    device=self.device
                )

                load_time = time.time() - start_time
                print(f"✅ Reranker 加载完成 ({load_time:.1f}秒, {self.device})")
            except Exception as e:
                print(f"⚠️ Reranker 加载失败: {e}")
                self.enable_reranker = False

    def _detect_device(self) -> str:
        """检测最佳计算设备"""
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        elif torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        else:
            return "cpu"

    def vector_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """向量检索"""
        import ollama

        # 生成查询向量
        response = ollama.embeddings(model="bge-m3", prompt=query)
        query_embedding = response["embedding"]

        # ChromaDB 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # 格式化结果
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'score': 1 / (1 + results['distances'][0][i]),
                'source': 'vector'
            })

        return formatted

    def bm25_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """BM25 检索"""
        # 中文分词
        tokenized_query = list(jieba.cut(query))

        # 计算 BM25 分数
        scores = self.bm25.get_scores(tokenized_query)

        # 获取 Top-K
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # 格式化结果
        formatted = []
        for idx in top_indices:
            formatted.append({
                'id': self.bm25_ids[idx],
                'document': self.bm25_documents[idx],
                'metadata': self.bm25_metadatas[idx],
                'score': scores[idx],
                'source': 'bm25'
            })

        return formatted

    def reciprocal_rank_fusion(self, results_list: List[List[Dict]], k: int = 60, max_candidates: int = 15) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 算法
        融合多路检索结果，限制最大候选数
        """
        doc_scores = {}

        for results in results_list:
            for rank, item in enumerate(results):
                doc_key = item['document']

                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {
                        'id': item['id'],
                        'document': item['document'],
                        'metadata': item['metadata'],
                        'rrf_score': 0,
                        'sources': []
                    }

                doc_scores[doc_key]['rrf_score'] += 1 / (k + rank + 1)
                doc_scores[doc_key]['sources'].append(item['source'])

        # 按 RRF 分数排序，限制候选数量
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['rrf_score'], reverse=True)

        return sorted_docs[:max_candidates]

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """Reranker 精排 (带计时 + 权重加权)"""
        if not self.enable_reranker or len(candidates) == 0:
            return candidates[:top_k]

        start_time = time.time()

        # 构造 query-doc 对
        pairs = [(query, item['document']) for item in candidates]

        # 计算相关性分数
        scores = self.reranker.predict(pairs)

        # 按分数重新排序（加入权重）
        reranked = []
        for i, item in enumerate(candidates):
            rerank_score = float(scores[i])
            weight = item.get('metadata', {}).get('weight', 0.5)

            # 最终分数 = rerank分数 * 权重放大因子
            weight_boost = 1 + weight
            final_score = rerank_score * weight_boost

            item['rerank_score'] = rerank_score
            item['weight'] = weight
            item['final_score'] = final_score
            reranked.append(item)

        # 按最终分数排序
        reranked.sort(key=lambda x: x['final_score'], reverse=True)

        elapsed = time.time() - start_time
        print(f"  ⚡ Reranker 耗时: {elapsed:.2f}秒 ({len(candidates)} 条)")

        return reranked[:top_k]

    def search(self, query: str, top_k: int = 5, use_reranker=True) -> List[Dict]:
        """
        混合检索主入口
        流程：向量检索 + BM25 → RRF 融合 → Reranker 精排
        """
        total_start = time.time()

        # 1. 向量检索
        t0 = time.time()
        vector_results = self.vector_search(query, top_k=VECTOR_TOP_K)
        print(f"  📊 向量检索: {len(vector_results)} 条 ({time.time()-t0:.2f}秒)")

        # 2. BM25 检索
        if self.enable_bm25:
            t0 = time.time()
            bm25_results = self.bm25_search(query, top_k=BM25_TOP_K)
            print(f"  📊 BM25 检索: {len(bm25_results)} 条 ({time.time()-t0:.2f}秒)")

            # 3. RRF 融合
            merged = self.reciprocal_rank_fusion(
                [vector_results, bm25_results],
                max_candidates=RERANKER_CANDIDATES
            )
            print(f"  🔀 RRF 融合: {len(merged)} 条")
        else:
            merged = vector_results[:RERANKER_CANDIDATES]

        # 4. Reranker 精排
        if use_reranker and self.enable_reranker:
            final = self.rerank(query, merged, top_k=top_k)
        else:
            final = merged[:top_k]

        total_time = time.time() - total_start
        print(f"  ⏱️ 总耗时: {total_time:.2f}秒")

        return final

# ============== 测试代码 ==============
if __name__ == "__main__":
    print("=" * 60)
    print("🔀 混合检索引擎测试")
    print("=" * 60)

    retriever = HybridRetriever(enable_reranker=True, enable_bm25=True)

    test_queries = [
        "我的个人网站是什么",
        "RAG 系统如何优化",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 查询: {query}")
        print(f"{'='*60}")

        results = retriever.search(query, top_k=3)

        for i, item in enumerate(results):
            print(f"\n [{i+1}] 来源: {item.get('sources', [item.get('source')])}")
            if 'rerank_score' in item:
                print(f"     Rerank: {item['rerank_score']:.4f}")
            if 'rrf_score' in item:
                print(f"     RRF: {item['rrf_score']:.4f}")
            print(f"     内容: {item['document'][:100]}...")
