#!/usr/bin/env python3
"""
BM25 索引构建器
从 ChromaDB 读取所有文档，构建 BM25 索引
"""

import os
import pickle
import jieba
from rank_bm25 import BM25Okapi
import chromadb
from tqdm import tqdm

# 配置
DB_PATH = os.path.expanduser("~/RAG_Memory_System/chroma_db")
BM25_INDEX_PATH = os.path.expanduser("~/RAG_Memory_System/bm25_index.pkl")

def build_bm25_index():
    """构建 BM25 索引"""
    print("=" * 60)
    print("🔨 BM25 索引构建器")
    print("=" * 60)

    # 连接 ChromaDB
    print("📂 连接向量数据库...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection("memory_v1_semantic")

    total = collection.count()
    print(f"📊 总文档数: {total:,}")

    # 分批读取所有文档
    print("📖 读取文档...")
    documents = []
    metadatas = []
    ids = []

    batch_size = 1000
    offset = 0

    while offset < total:
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"]
        )

        documents.extend(results['documents'])
        metadatas.extend(results['metadatas'])
        ids.extend(results['ids'])

        offset += batch_size
        print(f"  已读取: {len(documents):,} / {total:,}")

    print(f"✅ 读取完成: {len(documents):,} 条文档")

    # 中文分词
    print("🔪 分词中...")
    tokenized_docs = []

    for doc in tqdm(documents, desc="分词"):
        tokens = list(jieba.cut(doc))
        tokenized_docs.append(tokens)

    # 构建 BM25 索引
    print("🔨 构建 BM25 索引...")
    bm25 = BM25Okapi(tokenized_docs)

    # 保存索引
    print("💾 保存索引...")
    index_data = {
        'bm25': bm25,
        'documents': documents,
        'metadatas': metadatas,
        'ids': ids
    }

    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(index_data, f)

    # 检查文件大小
    file_size = os.path.getsize(BM25_INDEX_PATH) / (1024 * 1024)

    print("=" * 60)
    print(f"✅ BM25 索引构建完成!")
    print(f"   文件: {BM25_INDEX_PATH}")
    print(f"   大小: {file_size:.1f} MB")
    print(f"   文档数: {len(documents):,}")
    print("=" * 60)

if __name__ == "__main__":
    build_bm25_index()
