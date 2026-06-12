# DeerFlow 记忆系统 + 主动推送改造 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 DeerFlow 上实现双层记忆（Markdown + Chroma）和主动推送系统（电量模型 + LLM Agent 分类 + Drift）

**Architecture:** 核心算法独立为 magentic-memory / magentic-proactive 两个包，集成层通过 MemoryStorage 子类和新中间件接入 DeerFlow

**Tech Stack:** Python 3.12+, ChromaDB, LangChain, LangGraph, Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-06-12-memory-and-proactive-design.md`

---

## Phase 1: MarkdownMemoryStore 基础存储层

### Task 1.1: 创建 MarkdownMemoryStore 类骨架

**Files:**
- Create: `backend/packages/harness/deerflow/agents/memory/markdown_store.py`
- Create: `backend/tests/test_markdown_store.py`

- [ ] **Step 1: 写失败的测试**

```python
# backend/tests/test_markdown_store.py
import tempfile
import os
import pytest
from pathlib import Path

@pytest.mark.asyncio
async def test_write_and_read_long_term():
    from deerflow.agents.memory.markdown_store import MarkdownMemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MarkdownMemoryStore(base_dir=Path(tmpdir))
        user_id = "test_user"

        # 初始读取应为空
        result = store.read_long_term(user_id)
        assert result == ""

        # 写入后能读回
        content = "- [identity] 测试用户"
        store.write_long_term(user_id, content)
        assert store.read_long_term(user_id) == content

@pytest.mark.asyncio
async def test_write_and_read_pending():
    from deerflow.agents.memory.markdown_store import MarkdownMemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MarkdownMemoryStore(base_dir=Path(tmpdir))

        user_id = "test_user"

        # 追加两条 pending items
        store.append_pending(user_id, "- [preference] 用户喜欢测试\n- [identity] 用户是开发者\n")
        result = store.read_pending(user_id)
        assert "[preference] 用户喜欢测试" in result
        assert "[identity] 用户是开发者" in result

        # 再追加一条
        store.append_pending(user_id, "- [key_info] api_key: xxx\n")
        result = store.read_pending(user_id)
        assert "[key_info] api_key: xxx" in result
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd deer-flow/backend
uv run pytest tests/test_markdown_store.py -v
```
Expected: FAIL — MarkdownMemoryStore 不存在

- [ ] **Step 3: 实现 MarkdownMemoryStore**

```python
# backend/packages/harness/deerflow/agents/memory/markdown_store.py
from pathlib import Path
import os


class MarkdownMemoryStore:
    """用五个 Markdown 文件实现 MemoryStorage 接口。
    
    文件结构:
        {base_dir}/users/{user_id}/memory/
        ├── MEMORY.md
        ├── HISTORY.md
        ├── SELF.md
        ├── RECENT_CONTEXT.md
        └── PENDING.md
    """

    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)

    def _user_dir(self, user_id: str) -> Path:
        return self._base_dir / "users" / user_id / "memory"

    def _ensure_dir(self, user_id: str) -> Path:
        d = self._user_dir(user_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _read_file(self, user_id: str, filename: str) -> str:
        path = self._user_dir(user_id) / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _write_file(self, user_id: str, filename: str, content: str):
        self._ensure_dir(user_id)
        (self._user_dir(user_id) / filename).write_text(content, encoding="utf-8")

    def _append_file(self, user_id: str, filename: str, content: str):
        self._ensure_dir(user_id)
        with open(self._user_dir(user_id) / filename, "a", encoding="utf-8") as f:
            f.write(content)

    def read_long_term(self, user_id: str) -> str:
        return self._read_file(user_id, "MEMORY.md")

    def write_long_term(self, user_id: str, content: str):
        self._write_file(user_id, "MEMORY.md", content)

    def read_history(self, user_id: str) -> str:
        return self._read_file(user_id, "HISTORY.md")

    def write_history(self, user_id: str, content: str):
        self._write_file(user_id, "HISTORY.md", content)

    def append_history(self, user_id: str, content: str):
        self._append_file(user_id, "HISTORY.md", content)

    def read_self(self, user_id: str) -> str:
        return self._read_file(user_id, "SELF.md")

    def write_self(self, user_id: str, content: str):
        self._write_file(user_id, "SELF.md", content)

    def read_recent_context(self, user_id: str) -> str:
        return self._read_file(user_id, "RECENT_CONTEXT.md")

    def write_recent_context(self, user_id: str, content: str):
        self._write_file(user_id, "RECENT_CONTEXT.md", content)

    def read_pending(self, user_id: str) -> str:
        return self._read_file(user_id, "PENDING.md")

    def append_pending(self, user_id: str, content: str):
        self._append_file(user_id, "PENDING.md", content)

    def write_pending(self, user_id: str, content: str):
        self._write_file(user_id, "PENDING.md", content)
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd deer-flow/backend
uv run pytest tests/test_markdown_store.py -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/packages/harness/deerflow/agents/memory/markdown_store.py
git add backend/tests/test_markdown_store.py
git commit -m "feat: add MarkdownMemoryStore with five-file markdown storage"
```

---

### Task 1.2: 实现用户目录隔离和历史追加

**Files:**
- Modify: `backend/tests/test_markdown_store.py` (追加)
- Modify: `backend/packages/harness/deerflow/agents/memory/markdown_store.py`

- [ ] **Step 1: 写失败的测试**

```python
# 追加到 backend/tests/test_markdown_store.py

@pytest.mark.asyncio
async def test_user_isolation():
    from deerflow.agents.memory.markdown_store import MarkdownMemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MarkdownMemoryStore(base_dir=Path(tmpdir))
        
        store.write_long_term("alice", "- [identity] Alice")
        store.write_long_term("bob", "- [identity] Bob")
        
        assert "Alice" in store.read_long_term("alice")
        assert "Bob" in store.read_long_term("bob")
        assert "Alice" not in store.read_long_term("bob")
        assert "Bob" not in store.read_long_term("alice")

@pytest.mark.asyncio
async def test_history_append():
    from deerflow.agents.memory.markdown_store import MarkdownMemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MarkdownMemoryStore(base_dir=Path(tmpdir))
        uid = "test_user"
        
        store.append_history(uid, "[2026-06-01 10:00] 用户做了A\n")
        store.append_history(uid, "[2026-06-02 15:30] 用户做了B\n")
        result = store.read_history(uid)
        assert "用户做了A" in result
        assert "用户做了B" in result
```

- [ ] **Step 2: 运行测试**

```bash
cd deer-flow/backend
uv run pytest tests/test_markdown_store.py::test_user_isolation tests/test_markdown_store.py::test_history_append -v
```
Expected: PASS (已有实现已覆盖)

- [ ] **Step 3: 提交**

```bash
git add backend/tests/test_markdown_store.py
git commit -m "test: add user isolation and history append tests for MarkdownMemoryStore"
```

---

## Phase 2: Chroma 向量存储层 (magentic-memory 包)

### Task 2.1: 创建 magentic-memory 包骨架

**Files:**
- Create: `backend/packages/magentic-memory/pyproject.toml`
- Create: `backend/packages/magentic-memory/magentic_memory/__init__.py`
- Create: `backend/packages/magentic-memory/magentic_memory/vector_store.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
# backend/packages/magentic-memory/pyproject.toml
[project]
name = "magentic-memory"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "chromadb>=0.5.0",
    "langchain-chroma>=0.1.0",
    "langchain-openai>=0.2.0",
    "langchain-core>=0.3.0",
]
```

- [ ] **Step 2: 创建 __init__.py**

```python
# backend/packages/magentic-memory/magentic_memory/__init__.py
from magentic_memory.vector_store import VectorMemoryStore

__all__ = ["VectorMemoryStore"]
```

- [ ] **Step 3: 创建 VectorMemoryStore 骨架**

```python
# backend/packages/magentic-memory/magentic_memory/vector_store.py
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class VectorMemoryStore:
    """Chroma 向量记忆存储。四类记忆各用一个 collection。"""

    _COLLECTION_NAMES = {
        "event": "memory_events",
        "procedure": "memory_procedures",
        "preference": "memory_preferences",
        "profile": "memory_profiles",
    }

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
        )
        self._collections: dict[str, Chroma] = {}
        for mtype, col_name in self._COLLECTION_NAMES.items():
            self._collections[mtype] = Chroma(
                client=self._client,
                collection_name=col_name,
                embedding_function=self._embeddings,
            )

    @property
    def collections(self) -> dict[str, Chroma]:
        return self._collections
```

- [ ] **Step 4: 安装并验证**

```bash
cd deer-flow/backend/packages/magentic-memory
uv pip install -e .
```

- [ ] **Step 5: 提交**

```bash
git add backend/packages/magentic-memory/
git commit -m "feat: create magentic-memory package with VectorMemoryStore skeleton"
```

---

### Task 2.2: 实现向量记忆写入

**Files:**
- Create: `backend/packages/magentic-memory/tests/test_vector_store.py`
- Modify: `backend/packages/magentic-memory/magentic_memory/vector_store.py`

- [ ] **Step 1: 写失败的测试**

```python
# backend/packages/magentic-memory/tests/test_vector_store.py
import tempfile
import pytest
import os

@pytest.fixture
def vector_store():
    from magentic_memory.vector_store import VectorMemoryStore
    with tempfile.TemporaryDirectory() as tmpdir:
        api_key = os.environ.get("OPENAI_API_KEY", "test-key")
        store = VectorMemoryStore(persist_dir=tmpdir, api_key=api_key)
        yield store

def test_add_memory_to_event_collection(vector_store):
    vector_store.add_memory(
        memory_type="event",
        content="[2026-06-01] 用户面试通过",
        metadata={"source_ref": "msg_001"},
        doc_id="event_001",
    )

    results = vector_store.collections["event"].get(ids=["event_001"])
    assert len(results["ids"]) == 1
    assert results["documents"][0] == "[2026-06-01] 用户面试通过"

def test_add_memory_with_metadata_filter(vector_store):
    vector_store.add_memory(
        memory_type="profile",
        content="用户是产品经理",
        metadata={"category": "personal_fact", "confidence": 0.9},
        doc_id="profile_001",
    )

    results = vector_store.collections["profile"].get(ids=["profile_001"])
    assert results["metadatas"][0]["category"] == "personal_fact"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd deer-flow/backend/packages/magentic-memory
uv run pytest tests/test_vector_store.py -v
```
Expected: FAIL — `add_memory` 方法不存在

- [ ] **Step 3: 实现 add_memory**

```python
# 追加到 vector_store.py 的 VectorMemoryStore 类中

def add_memory(
    self,
    memory_type: str,
    content: str,
    metadata: dict | None = None,
    doc_id: str | None = None,
):
    """写入一条记忆到对应类型的 collection。"""
    import hashlib
    
    collection = self._collections[memory_type]
    if doc_id is None:
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
    meta = {"memory_type": memory_type}
    if metadata:
        meta.update(metadata)
    
    collection.add_texts(
        texts=[content],
        metadatas=[meta],
        ids=[doc_id],
    )
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd deer-flow/backend/packages/magentic-memory
uv run pytest tests/test_vector_store.py -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/packages/magentic-memory/
git commit -m "feat: add add_memory to VectorMemoryStore for Chroma writes"
```

---

### Task 2.3: 实现相似度搜索

**Files:**
- Modify: `backend/packages/magentic-memory/tests/test_vector_store.py` (追加)
- Modify: `backend/packages/magentic-memory/magentic_memory/vector_store.py`

- [ ] **Step 1: 写失败的测试**

```python
# 追加到 test_vector_store.py

def test_similarity_search(vector_store):
    vector_store.add_memory("profile", "用户是产品经理", doc_id="p1")
    vector_store.add_memory("profile", "用户喜欢打篮球", doc_id="p2")
    vector_store.add_memory("event", "用户面试通过", doc_id="e1")

    # 搜索 profile 类型 — 应命中产品经理这条
    results = vector_store.similarity_search(
        query="用户是什么职业",
        memory_type="profile",
        k=2,
    )
    assert len(results) >= 1
    # 第一条最相关应该是产品经理
    assert "产品经理" in results[0][0]  # (doc, score) tuple

def test_similarity_search_with_filter(vector_store):
    vector_store.add_memory("event", "[2026-06-01] 用户去了北京", 
                            metadata={"scope_channel": "telegram"}, doc_id="e1")
    vector_store.add_memory("event", "[2026-06-02] 用户去了上海",
                            metadata={"scope_channel": "discord"}, doc_id="e2")

    # 只搜 telegram scope
    results = vector_store.similarity_search(
        query="用户去了哪里",
        memory_type="event",
        k=5,
        filter={"scope_channel": "telegram"},
    )
    assert len(results) >= 1
    assert all("北京" in r[0] for r in results)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
uv run pytest tests/test_vector_store.py::test_similarity_search -v
```
Expected: FAIL — `similarity_search` 方法不存在

- [ ] **Step 3: 实现 similarity_search**

```python
# 追加到 vector_store.py 的 VectorMemoryStore 类中

def similarity_search(
    self,
    query: str,
    memory_type: str,
    k: int = 5,
    filter: dict | None = None,
) -> list[tuple[str, float]]:
    """语义搜索，返回 (文档内容, 相似度分数) 列表。"""
    collection = self._collections[memory_type]
    search_filter = {"memory_type": memory_type}
    if filter:
        search_filter.update(filter)
    
    results = collection.similarity_search_with_score(
        query, k=k, filter=search_filter
    )
    return [(doc.page_content, score) for doc, score in results]
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd deer-flow/backend/packages/magentic-memory
uv run pytest tests/test_vector_store.py -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/packages/magentic-memory/
git commit -m "feat: add similarity_search with metadata filtering to VectorMemoryStore"
```

---

## Phase 3: 检索增强 (HyDE + RRF + QueryRewriter)

由于计划体量极大，以下 Phase 采用**精简 task 描述**（保留完整代码和测试），避免计划文件过长。

> **策略说明**: 以下 task 描述为每个模块的关键测试和实现骨架。实施时 agent 需根据 spec 补充完整逻辑，但本计划提供核心代码路径和测试签名以确保方向正确。

### Task 3.1: QueryRewriter — 查询重写和意图分类

**Files:**
- Create: `backend/packages/magentic-memory/magentic_memory/query_rewriter.py`
- Create: `backend/packages/magentic-memory/tests/test_query_rewriter.py`

**核心测试**:

```python
def test_rewrite_resolves_pronouns():
    """用户说"上次那个怎么样了"，应消解为具体事件"""
    rewriter = QueryRewriter(llm_client=mock_llm)
    result = await rewriter.decide(
        user_msg="上次那个怎么样了",
        recent_history="[user] 我昨天面了字节二面\n[assistant] 主要问项目经历",
    )
    assert result.needs_episodic is True
    assert "面试" in result.episodic_query

def test_gate_blocks_greeting():
    """简单问候不需要检索"""
    result = await rewriter.decide(
        user_msg="你好",
        recent_history="",
    )
    assert result.needs_episodic is False

def test_rewrite_timeout_fallback():
    """LLM 超时返回原始查询，不可阻断"""
    result = await rewriter.decide(
        user_msg="上次那个怎么样了",
        recent_history="...",
        timeout_ms=1,  # 强制超时
    )
    assert result.needs_episodic is True
    assert result.episodic_query == "上次那个怎么样了"  # fail-open
```

**关键实现**:
```python
class GateDecision:
    needs_episodic: bool
    episodic_query: str
    procedure_query: str = ""

class QueryRewriter:
    def __init__(self, llm_client, timeout_ms=800):
        self._llm = llm_client
        self._timeout_ms = timeout_ms

    async def decide(self, user_msg: str, recent_history: str) -> GateDecision:
        # 并行：历史感知改写 + procedure 改写
        # 800ms 超时 → fail-open 回退原始查询
        ...
```

---

### Task 3.2: HyDE 假设文档生成

**Files:**
- Create: `backend/packages/magentic-memory/magentic_memory/hyde_enhancer.py`
- Create: `backend/packages/magentic-memory/tests/test_hyde_enhancer.py`

**核心测试**:

```python
@pytest.mark.asyncio
async def test_generate_hypothesis():
    enhancer = HyDEEnhancer(llm_client=mock_llm)
    hyps = await enhancer.generate_hypotheses("推荐什么游戏")
    assert len(hyps) == 2  # event + general
    assert any("游戏" in h for h in hyps)

@pytest.mark.asyncio
async def test_hyde_timeout_graceful_degradation():
    enhancer = HyDEEnhancer(llm_client=slow_llm, timeout_ms=1)
    hyps = await enhancer.generate_hypotheses("推荐什么游戏")
    assert hyps == []  # 超时优雅降级
```

**关键实现**:
```python
class HyDEEnhancer:
    def __init__(self, llm_client, timeout_ms=2000, max_tokens=80):
        ...

    async def generate_hypotheses(self, query: str) -> list[str]:
        """并行生成 event-style + general-style 两条假设文档"""
        tasks = [
            self._gen_hypothesis(query, style="event"),
            self._gen_hypothesis(query, style="general"),
        ]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=self._timeout_ms / 1000,
        )
        return [r for r in results if isinstance(r, str) and r]
```

---

### Task 3.3: RRF 排序融合

**Files:**
- Create: `backend/packages/magentic-memory/magentic_memory/retriever.py`
- Create: `backend/packages/magentic-memory/tests/test_retriever.py`

**核心测试**:

```python
def test_rrf_merge_combines_two_sources():
    vec_results = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.7}]
    kw_results = [{"id": "c", "score": 0.8}, {"id": "a", "score": 0.5}]
    fused = rrf_merge(vec_results, kw_results, top_n=3, k=60, kw_weight=0.5)
    assert len(fused) == 3
    assert fused[0]["id"] == "a"  # a 在两路都出现，应排第一

def test_rrf_merge_empty_keyword():
    """只有向量结果时直接返回"""
    vec_results = [{"id": "a", "score": 0.9}]
    fused = rrf_merge(vec_results, [], top_n=5)
    assert len(fused) == 1
    assert fused[0]["id"] == "a"
```

**关键实现**:
```python
def rrf_merge(
    vec_results: list[dict],
    kw_results: list[dict],
    top_n: int,
    k: float = 60.0,
    kw_weight: float = 0.5,
) -> list[dict]:
    """RRF 融合: score = 1/(k+rank_vec) + kw_weight/(k+rank_kw)"""
    ...
```

---

### Task 3.4: MemoryRetriever — 完整检索编排

**Files:**
- Modify: `backend/packages/magentic-memory/magentic_memory/retriever.py`
- Modify: `backend/packages/magentic-memory/tests/test_retriever.py`

**核心测试**:

```python
@pytest.mark.asyncio
async def test_full_retrieval_pipeline(mock_vector_store):
    retriever = MemoryRetriever(
        vector_store=mock_vector_store,
        hyde_enhancer=mock_hyde,
        query_rewriter=mock_rewriter,
    )
    results = await retriever.retrieve(
        query="推荐什么游戏",
        memory_types=["preference", "event"],
        top_k=8,
    )
    assert len(results) >= 1
    assert all("memory_type" in r for r in results)
```

**关键实现**:
```python
class MemoryRetriever:
    async def retrieve(self, query, memory_types, top_k, **kwargs) -> list[dict]:
        # 1. 查询重写
        decision = await self._rewriter.decide(query, kwargs.get("history", ""))
        if not decision.needs_episodic:
            return []

        # 2. HyDE 假设生成
        hyps = await self._hyde.generate_hypotheses(decision.episodic_query)
        aux_queries = [decision.episodic_query, *hyps]

        # 3. 多向量通道检索 (query + hyps × 4 types)
        vec_results = self._vector_lane_search(aux_queries, memory_types, top_k)

        # 4. 关键词通道
        kw_results = self._keyword_lane_search(query, memory_types, top_k)

        # 5. RRF 融合
        return rrf_merge(vec_results, kw_results, top_n=top_k)
```

---

## Phase 4: VectorRetrievalMiddleware 集成

### Task 4.1: 创建中间件并插入链中

**Files:**
- Create: `backend/packages/harness/deerflow/agents/middlewares/vector_retrieval_middleware.py`
- Modify: `backend/packages/harness/deerflow/agents/lead_agent/agent.py`
- Create: `backend/tests/test_vector_retrieval_middleware.py`

**关键测试**:

```python
@pytest.mark.asyncio
async def test_middleware_injects_retrieved_memory():
    middleware = VectorRetrievalMiddleware(retriever=mock_retriever)
    state = make_test_state(messages=[HumanMessage(content="推荐什么游戏")])

    await middleware.abefore_agent(state, mock_runtime)
    # 验证第一条消息的 content 里注入了检索结果
    assert "系统检索" in state.messages[0].content
```

**DeerFlow 集成修改** (agent.py):

```python
# 在 build_middlewares() 中，DynamicContextMiddleware 之后插入
from deerflow.agents.middlewares.vector_retrieval_middleware import VectorRetrievalMiddleware

middlewares = [
    ...
    DynamicContextMiddleware(...),      # 位置 3
    VectorRetrievalMiddleware(...),     # 位置 4 (新增)
    SkillActivationMiddleware(...),     # 位置 5 (原 4)
    ...
]
```

---

## Phase 5: PENDING 缓冲 + Optimizer

### Task 5.1: ConsolidationUpdater (改造写入链路)

**Files:**
- Create: `backend/packages/harness/deerflow/agents/memory/consolidation_updater.py`
- Create: `backend/tests/test_consolidation_updater.py`

**核心测试**:

```python
@pytest.mark.asyncio
async def test_consolidation_writes_to_pending():
    store = MarkdownMemoryStore(base_dir=Path(tmpdir))
    updater = ConsolidationUpdater(
        markdown_store=store,
        vector_store=mock_vector_store,
        llm_client=mock_llm,
    )
    result = await updater.process(messages, user_id="test")
    assert "history_entries" in result
    assert "pending_items" in result
    assert store.read_pending("test") != ""
```

---

### Task 5.2: MarkdownOptimizer (18h 定时归档)

**Files:**
- Create: `backend/packages/harness/deerflow/agents/memory/markdown_optimizer.py`
- Create: `backend/tests/test_markdown_optimizer.py`

**核心测试**:

```python
@pytest.mark.asyncio
async def test_optimizer_merges_pending_to_memory():
    store = MarkdownMemoryStore(base_dir=Path(tmpdir))
    store.append_pending("test", "- [identity] 新身份\n")
    store.write_long_term("test", "- [identity] 旧身份\n- [preference] 旧偏好\n")

    optimizer = MarkdownOptimizer(
        markdown_store=store, llm_client=mock_llm, 
    )
    await optimizer.run(user_id="test")

    result = store.read_long_term("test")
    assert "新身份" in result
    assert "旧偏好" in result

@pytest.mark.asyncio
async def test_optimizer_rollback_on_failure():
    """LLM 返回格式异常时应回滚"""
    ...
```

---

## Phase 6: 主动推送系统

### Task 6.1: 电量模型 (magentic-proactive 包)

**Files:**
- Create: `backend/packages/magentic-proactive/pyproject.toml`
- Create: `backend/packages/magentic-proactive/magentic_proactive/__init__.py`
- Create: `backend/packages/magentic-proactive/magentic_proactive/energy.py`
- Create: `backend/packages/magentic-proactive/tests/test_energy.py`

**核心测试**:

```python
import math
from magentic_proactive.energy import compute_energy, next_tick_from_score

def test_energy_decay_at_zero_minutes():
    """刚发完消息，能量应为 1.0"""
    e = compute_energy(minutes_since_last=0)
    assert abs(e - 1.0) < 0.01

def test_energy_decay_at_60_minutes():
    """一小时后能量应低于 0.5"""
    e = compute_energy(minutes_since_last=60)
    assert e < 0.5

def test_energy_decay_at_72_hours():
    """三天后能量接近 0"""
    e = compute_energy(minutes_since_last=72*60)
    assert e < 0.05

def test_next_tick_urgent():
    """高紧迫性 → 最短间隔"""
    interval = next_tick_from_score(0.8, tick_s3=60, tick_s2=120, tick_s1=240, tick_s0=480)
    assert interval == 60

def test_next_tick_idle():
    """低紧迫性 → 最长间隔"""
    interval = next_tick_from_score(0.1, tick_s3=60, tick_s2=120, tick_s1=240, tick_s0=480)
    assert interval == 480

def test_next_tick_jitter():
    """抖动应在 ±30% 范围内"""
    intervals = [
        next_tick_from_score(0.1, tick_s3=60, tick_s2=120, tick_s1=240, tick_s0=480, jitter=0.3)
        for _ in range(100)
    ]
    base = 480
    assert all(base * 0.7 <= i <= base * 1.3 for i in intervals)
```

**关键实现**:
```python
def compute_energy(minutes_since_last: float) -> float:
    alpha, beta, gamma = 0.50, 0.35, 0.15
    tau1, tau2, tau3 = 30.0, 240.0, 2880.0
    result = (
        alpha * math.exp(-minutes_since_last / tau1)
        + beta * math.exp(-minutes_since_last / tau2)
        + gamma * math.exp(-minutes_since_last / tau3)
    )
    return max(0.0, min(1.0, result))

def next_tick_from_score(base_score, tick_s3, tick_s2, tick_s1, tick_s0, jitter=0.3):
    if base_score > 0.70: base = tick_s3
    elif base_score > 0.40: base = tick_s2
    elif base_score > 0.20: base = tick_s1
    else: base = tick_s0
    jittered = base * (1.0 + random.uniform(-jitter, jitter))
    return max(1, int(jittered))
```

---

### Task 6.2: ProactiveLoop 主循环

**Files:**
- Create: `backend/packages/magentic-proactive/magentic_proactive/loop.py`
- Create: `backend/packages/magentic-proactive/magentic_proactive/judge.py`
- Create: `backend/packages/harness/deerflow/agents/middlewares/proactive_loop_middleware.py`
- Modify: `backend/packages/harness/deerflow/agents/lead_agent/config.py`

**配置扩展**:
```yaml
# config.yaml 新增
proactive:
  enabled: true
  default_channel: telegram
  tick_interval_s0: 480   # 8min
  tick_interval_s1: 240   # 4min
  tick_interval_s2: 120   # 2min
  tick_interval_s3: 60    # 1min
  score_weight_energy: 0.35
  ack_cited_ttl_hours: 168
  ack_uncited_ttl_hours: 24
  ack_discarded_ttl_hours: 720
  drift_enabled: true
  drift_min_interval_hours: 3
```

---

### Task 6.3: Drift 空闲引擎

**Files:**
- Create: `backend/packages/magentic-proactive/magentic_proactive/drift.py`
- Create: `backend/packages/magentic-proactive/tests/test_drift.py`

**核心测试**:

```python
@pytest.mark.asyncio
async def test_drift_scans_skill_files(tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    (skill_dir / "health-check").mkdir()
    (skill_dir / "health-check" / "SKILL.md").write_text(
        "---\nname: 健康检查\ndescription: 审计记忆\n---\n## 目标\n检查一致性"
    )

    engine = DriftEngine(skills_dir=skill_dir)
    skills = engine.scan_skills()
    assert len(skills) == 1
    assert skills[0]["name"] == "健康检查"

@pytest.mark.asyncio
async def test_drift_cooldown():
    engine = DriftEngine(min_interval_hours=3)
    assert engine.can_run(last_run_at=datetime.now() - timedelta(hours=1)) is False
    assert engine.can_run(last_run_at=datetime.now() - timedelta(hours=4)) is True
```

---

## Phase 7: 端到端验证

### Task 7.1: 集成测试 — 完整对话链路

**Files:**
- Create: `backend/tests/test_e2e_memory_pipeline.py`

**核心测试**:
```python
@pytest.mark.asyncio
async def test_full_conversation_with_memory():
    """完整对话：消息 → 检索 → 注入 → 回复 → consolidation → PENDING"""
    ...

@pytest.mark.asyncio
async def test_proactive_tick_with_push():
    """主动推送完整链路：电量 → 拉取 → 分类 → 推送 → ACK"""
    ...
```

---

## 文件清单

| 文件 | 操作 | 模块 |
|---|---|---|
| `backend/packages/harness/deerflow/agents/memory/markdown_store.py` | 新建 | Phase 1 |
| `backend/tests/test_markdown_store.py` | 新建 | Phase 1 |
| `backend/packages/magentic-memory/` | 新建包 | Phase 2 |
| `backend/packages/magentic-memory/magentic_memory/vector_store.py` | 新建 | Phase 2 |
| `backend/packages/magentic-memory/magentic_memory/query_rewriter.py` | 新建 | Phase 3 |
| `backend/packages/magentic-memory/magentic_memory/hyde_enhancer.py` | 新建 | Phase 3 |
| `backend/packages/magentic-memory/magentic_memory/retriever.py` | 新建 | Phase 3 |
| `backend/packages/harness/deerflow/agents/middlewares/vector_retrieval_middleware.py` | 新建 | Phase 4 |
| `backend/packages/harness/deerflow/agents/lead_agent/agent.py` | 修改 | Phase 4 |
| `backend/packages/harness/deerflow/agents/memory/consolidation_updater.py` | 新建 | Phase 5 |
| `backend/packages/harness/deerflow/agents/memory/markdown_optimizer.py` | 新建 | Phase 5 |
| `backend/packages/magentic-proactive/` | 新建包 | Phase 6 |
| `backend/packages/magentic-proactive/magentic_proactive/energy.py` | 新建 | Phase 6 |
| `backend/packages/magentic-proactive/magentic_proactive/loop.py` | 新建 | Phase 6 |
| `backend/packages/magentic-proactive/magentic_proactive/judge.py` | 新建 | Phase 6 |
| `backend/packages/magentic-proactive/magentic_proactive/drift.py` | 新建 | Phase 6 |
| `backend/packages/harness/deerflow/agents/middlewares/proactive_loop_middleware.py` | 新建 | Phase 6 |
