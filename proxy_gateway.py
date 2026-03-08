#!/usr/bin/env python3
"""
🧠 记忆注入代理网关 - LLM API 适配版 (Phase 2.0 数字分身版)

把记忆注入到用户消息的开头
Phase 2.0 更新：
- 身份认同：从"AI助手"升级为"数字分身"，第一人称视角
- 多元信息源：身份锚点 + 检索结果 + 推理，而非盲从检索
- 角色辨别：防止把聊天记录中的他人信息当成自己的
- 物理缴械：移除本地搜索工具防止模型擅自调用
"""

import json
import copy
import urllib.request
import urllib.error
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import uvicorn

# ============== 配置 ==============
# 根据你的环境修改以下配置
REAL_API_BASE = "https://api.your-llm-provider.com"  # 你的 LLM API 地址
HIPPOCAMPUS_URL = "http://localhost:8000/search"
PROXY_PORT = 8080

# 需要被缴械的工具列表（这些工具会在本地文件系统搜索，与 RAG 冲突）
DISABLED_TOOLS = ["search_memories", "read_file", "search_files"]

# ============== 新版 Prompt（Phase 2.0 数字分身版）==============
# ⚠️ 请根据你的个人信息修改以下内容
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

# ============== 海马体检索 ==============
def search_memory_sync(query: str, top_k: int = 3) -> str:
    """向海马体检索相关记忆"""
    try:
        data = json.dumps({"query": query, "top_k": top_k}).encode('utf-8')
        req = urllib.request.Request(
            HIPPOCAMPUS_URL,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        response = urllib.request.urlopen(req, timeout=60)
        result = json.loads(response.read().decode('utf-8'))

        if result.get("status") == "success":
            memories = result.get("memories", [])
            if memories:
                context = ""
                for i, m in enumerate(memories, 1):
                    content = m['content'][:500] + "..." if len(m['content']) > 500 else m['content']
                    context += f"\n[来源: {m['source']}]\n{content}\n"
                print(f"[代理] ✅ 注入 {len(memories)} 条记忆")
                return context
    except urllib.error.URLError:
        print("[代理] ⚠️ 海马体服务未响应")
    except Exception as e:
        print(f"[代理] ❌ 海马体查询失败: {e}")
    return ""

# ============== 提取用户问题 ==============
def extract_user_message(body: dict) -> str:
    """从请求体中提取用户最新的一条消息"""
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        return item.get("text", "")
            elif isinstance(content, str):
                return content
    return ""

# ============== 注入记忆（Phase 2.0 数字分身版）==============
def inject_memory_to_request(body: dict, user_question: str, memory_context: str) -> dict:
    """
    把记忆注入到第一条 user 消息的开头
    Phase 2.0：数字分身版 - 多元信息源 + 角色辨别 + 推理边界
    """
    body = copy.deepcopy(body)

    # 构建新版注入内容
    if memory_context:
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
    else:
        rag_injection = """<answering_rules>
当前没有相关记忆切片。优先基于身份锚点回答，否则说"这个我得查一下"。

【安全边界】
- 禁止调用任何工具验证信息，直接基于已有信息回答
</answering_rules>"""

    # 完整的上下文前缀
    full_prefix = f"""{BASE_IDENTITY}

{rag_injection}

"""

    messages = body.get("messages", [])

    # 找到最新一条 user 消息，在其内容前注入上下文
    injected = False
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = full_prefix + content
                injected = True
                print(f"[代理] 💉 注入成功 (string格式)")
            elif isinstance(content, list):
                # 处理多模态消息 - 注入到最后一个 text 元素
                last_text_idx = -1
                for i, item in enumerate(content):
                    if item.get("type") == "text":
                        last_text_idx = i

                if last_text_idx >= 0:
                    original_len = len(content[last_text_idx].get("text", ""))
                    content[last_text_idx]["text"] = full_prefix + content[last_text_idx].get("text", "")
                    injected = True
                    print(f"[代理] 💉 注入成功 (多模态格式, 第{last_text_idx}个text元素)")
            break

    if not injected:
        print(f"[代理] ⚠️ 注入失败！未找到可注入的文本内容")

    body["messages"] = messages
    return body

# ============== 物理缴械：移除冲突工具 ==============
def disarm_conflicting_tools(body: dict) -> dict:
    """
    Phase 1.5 核心功能：物理缴械
    在发送给 LLM 之前，移除与 RAG 冲突的本地搜索工具
    """
    if "tools" in body and isinstance(body["tools"], list):
        original_count = len(body["tools"])
        allowed_tools = []

        for tool in body["tools"]:
            tool_name = ""
            if isinstance(tool, dict):
                if "function" in tool and isinstance(tool["function"], dict):
                    tool_name = tool["function"].get("name", "")
                else:
                    tool_name = tool.get("name", "")

            if tool_name not in DISABLED_TOOLS:
                allowed_tools.append(tool)
            else:
                print(f"[代理] 🔒 缴械工具: {tool_name}")

        body["tools"] = allowed_tools
        if original_count != len(allowed_tools):
            print(f"[代理] 🛡️ 工具过滤: {original_count} → {len(allowed_tools)}")

    return body

# ============== FastAPI 应用 ==============
app = FastAPI(title="🧠 记忆注入代理 (Phase 2.0 数字分身版)")

@app.get("/")
def root():
    return {
        "service": "🧠 记忆注入代理网关",
        "version": "Phase 2.0 数字分身版",
        "status": "running",
        "hippocampus": HIPPOCAMPUS_URL,
        "target": REAL_API_BASE,
        "disabled_tools": DISABLED_TOOLS,
        "features": ["数字分身", "身份锚点", "角色辨别", "推理边界"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/messages")
async def proxy_messages(request: Request):
    """代理 messages API 调用"""
    body_bytes = await request.body()

    try:
        body = json.loads(body_bytes)

        # 提取用户问题
        user_msg = extract_user_message(body)
        print(f"[代理] 📩 用户问题: {user_msg[:80]}...")

        # 查询海马体
        memory_context = ""
        if user_msg:
            # 降噪滤网：只取最后 300 字符进行向量检索
            search_query = user_msg[-300:] if len(user_msg) > 300 else user_msg
            if len(user_msg) > 300:
                print(f"[代理] 🔇 降噪截断: {len(user_msg)}字符 → 取最后300字符")
            memory_context = search_memory_sync(search_query)

        # Step 1: 注入记忆
        body = inject_memory_to_request(body, user_msg, memory_context)

        # Step 2: 物理缴械
        body = disarm_conflicting_tools(body)

        body_bytes = json.dumps(body).encode('utf-8')
        print(f"[代理] 📤 转发请求到 LLM API...")

    except Exception as e:
        print(f"[代理] ❌ 处理失败: {e}")

    # 转发给 LLM API
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    url = f"{REAL_API_BASE}/v1/messages"

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, content=body_bytes, headers=headers)

        async def stream_response():
            async for chunk in response.aiter_bytes():
                yield chunk

        return StreamingResponse(
            stream_response(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/event-stream")
        )

# ============== 启动 ==============
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 记忆注入代理网关 (Phase 2.0 数字分身版)")
    print("=" * 60)
    print(f"  本地端口: {PROXY_PORT}")
    print(f"  海马体:   {HIPPOCAMPUS_URL}")
    print(f"  目标 API: {REAL_API_BASE}")
    print(f"  缴械工具: {DISABLED_TOOLS}")
    print(f"  新特性: 数字分身 | 身份锚点 | 角色辨别 | 推理边界")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
