#!/usr/bin/env python3
"""
Git 增量同步脚本 v5
功能：Git Pull → 检测变化 → MD5去重 → 权重计算 → 切片向量化入库
"""

import os
import sys
import json
import hashlib
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple

# 配置 - 根据你的环境修改
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
REPOS_CONFIG = os.path.join(CONFIG_DIR, "repos_config.json")
WEIGHT_RULES = os.path.join(CONFIG_DIR, "weight_rules.json")
MD5_CACHE = os.path.join(CONFIG_DIR, ".md5_cache.json")
DB_PATH = os.path.expanduser("~/RAG_Memory_System/chroma_db")

# 切片配置
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def load_json(path: str, default=None):
    """加载 JSON 文件"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default

def save_json(path: str, data):
    """保存 JSON 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_file_md5(filepath: str) -> str:
    """计算文件 MD5"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def run_git_pull(repo_path: str) -> bool:
    """执行 git pull"""
    try:
        result = subprocess.run(
            ['git', 'pull'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60
        )
        print(f"[Git] {result.stdout.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"[Git] 拉取失败: {e}")
        return False

def get_changed_files(repo_path: str, full_rebuild: bool = False) -> List[str]:
    """获取需要处理的文件"""
    if full_rebuild:
        # 全量重建：处理所有 .md 文件
        files = []
        for root, _, filenames in os.walk(repo_path):
            for filename in filenames:
                if filename.endswith('.md'):
                    files.append(os.path.join(root, filename))
        return files
    else:
        # 增量更新：只处理有变化的文件
        # 这里简化处理，实际可以用 git diff 获取
        files = []
        for root, _, filenames in os.walk(repo_path):
            for filename in filenames:
                if filename.endswith('.md'):
                    files.append(os.path.join(root, filename))
        return files

def calculate_weight(rel_path: str, weight_rules: dict) -> float:
    """根据路径计算权重"""
    rules = weight_rules.get("rules", {})
    default = weight_rules.get("default_weight", 0.5)

    for pattern, rule in rules.items():
        if pattern in rel_path:
            return rule.get("weight", default)

    return default

def split_markdown(content: str) -> List[str]:
    """双重语义切片"""
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter
    )

    # 第一重：按 Markdown 标题层级
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    try:
        md_splits = markdown_splitter.split_text(content)
    except:
        md_splits = [type('obj', (object,), {'page_content': content})]

    # 第二重：递归字符切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )

    chunks = []
    for split in md_splits:
        text = split.page_content if hasattr(split, 'page_content') else str(split)
        sub_chunks = text_splitter.split_text(text)
        chunks.extend(sub_chunks)

    return [c for c in chunks if len(c.strip()) > 20]

def index_file(collection, filepath: str, rel_path: str, repo_name: str, weight_rules: dict):
    """索引单个文件（批量写入）"""
    import ollama

    # 读取文件
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ⚠️ 读取失败: {e}")
        return

    if not content.strip():
        return

    # 计算权重
    weight = calculate_weight(rel_path, weight_rules)

    # 切片
    chunks = split_markdown(content)
    if not chunks:
        return

    print(f"  📄 {rel_path}: {len(chunks)} 个切片, 权重 {weight}")

    # 批量处理
    batch_ids = []
    batch_embeddings = []
    batch_documents = []
    batch_metadatas = []

    for idx, chunk in enumerate(chunks):
        try:
            # 向量化
            response = ollama.embeddings(model="bge-m3", prompt=chunk)
            embedding = response["embedding"]

            # 构建数据
            chunk_id = f"{repo_name}:{rel_path}:{idx}"
            batch_ids.append(chunk_id)
            batch_embeddings.append(embedding)
            batch_documents.append(chunk)
            batch_metadatas.append({
                "source": f"{repo_name}/{rel_path}",
                "weight": weight,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            })

        except Exception as e:
            print(f"    ⚠️ 切片 {idx} 失败: {e}")
            continue

    # 批量写入
    if batch_ids:
        try:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
        except Exception as e:
            print(f"    ❌ 写入失败: {e}")

def delete_old_vectors(collection, repo_name: str, rel_path: str):
    """删除文件的旧向量"""
    try:
        # 查询该文件的所有向量
        prefix = f"{repo_name}:{rel_path}:"
        results = collection.get(
            where={"source": f"{repo_name}/{rel_path}"},
            include=["metadatas"]
        )

        if results['ids']:
            collection.delete(ids=results['ids'])
            print(f"  🗑️ 删除旧向量: {len(results['ids'])} 条")

    except Exception as e:
        print(f"  ⚠️ 删除旧向量失败: {e}")

def main(full_rebuild: bool = False):
    """主函数"""
    import chromadb
    import ollama

    print("=" * 60)
    print("🔄 Git 记忆同步 v5")
    print("=" * 60)
    print(f"模式: {'全量重建' if full_rebuild else '增量更新'}")

    # 加载配置
    repos = load_json(REPOS_CONFIG, [])
    weight_rules = load_json(WEIGHT_RULES, {"rules": {}, "default_weight": 0.5})
    md5_cache = load_json(MD5_CACHE, {})

    if not repos:
        print("❌ 未找到仓库配置，请检查 repos_config.json")
        return

    # 连接数据库
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name="memory_v1_semantic",
        metadata={"description": "私有知识库"}
    )

    print(f"📊 当前向量数: {collection.count():,}")

    # 处理每个仓库
    for repo in repos:
        repo_name = repo['name']
        repo_path = repo['path']

        print(f"\n{'='*60}")
        print(f"📁 仓库: {repo_name}")
        print(f"📂 路径: {repo_path}")
        print("=" * 60)

        # Git Pull
        if not full_rebuild:
            print("🔄 执行 git pull...")
            run_git_pull(repo_path)

        # 获取文件列表
        files = get_changed_files(repo_path, full_rebuild)
        print(f"📄 待处理文件: {len(files)}")

        # 处理文件
        processed = 0
        skipped = 0

        for filepath in files:
            rel_path = os.path.relpath(filepath, repo_path)

            # MD5 检查（增量模式下）
            if not full_rebuild:
                current_md5 = get_file_md5(filepath)
                cached_md5 = md5_cache.get(filepath)

                if cached_md5 == current_md5:
                    skipped += 1
                    continue

                md5_cache[filepath] = current_md5

            # 删除旧向量
            delete_old_vectors(collection, repo_name, rel_path)

            # 索引文件
            index_file(collection, filepath, rel_path, repo_name, weight_rules)
            processed += 1

        print(f"\n✅ 处理完成: {processed} 个文件, 跳过 {skipped} 个未变化")

    # 保存 MD5 缓存
    save_json(MD5_CACHE, md5_cache)

    print(f"\n{'='*60}")
    print(f"🎉 同步完成！最终向量数: {collection.count():,}")
    print("=" * 60)

if __name__ == "__main__":
    full = "--full" in sys.argv
    main(full_rebuild=full)
