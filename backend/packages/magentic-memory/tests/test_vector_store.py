import os
import tempfile

import pytest

# 直接加载模块避开包依赖
import importlib.util
import sys
from pathlib import Path

_PKG = Path(__file__).parent.parent / "magentic_memory" / "vector_store.py"
_spec = importlib.util.spec_from_file_location("vector_store", _PKG)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["vector_store"] = _mod
_spec.loader.exec_module(_mod)
VectorMemoryStore = _mod.VectorMemoryStore


@pytest.fixture
def vector_store():
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


def test_add_memory_with_metadata_filter(vector_store):
    vector_store.add_memory(
        memory_type="profile",
        content="用户是产品经理",
        metadata={"category": "personal_fact", "confidence": 0.9},
        doc_id="profile_001",
    )

    results = vector_store.collections["profile"].get(ids=["profile_001"])
    assert results["metadatas"][0]["category"] == "personal_fact"


def test_similarity_search(vector_store):
    vector_store.add_memory("profile", "用户是产品经理", doc_id="p1")
    vector_store.add_memory("profile", "用户喜欢打篮球", doc_id="p2")
    vector_store.add_memory("event", "用户面试通过", doc_id="e1")

    results = vector_store.similarity_search(
        query="用户是什么职业",
        memory_type="profile",
        k=2,
    )
    assert len(results) >= 1


def test_similarity_search_with_filter(vector_store):
    vector_store.add_memory(
        "event", "[2026-06-01] 用户去了北京",
        metadata={"scope_channel": "telegram"}, doc_id="e1",
    )
    vector_store.add_memory(
        "event", "[2026-06-02] 用户去了上海",
        metadata={"scope_channel": "discord"}, doc_id="e2",
    )

    results = vector_store.similarity_search(
        query="用户去了哪里",
        memory_type="event",
        k=5,
        filter={"scope_channel": "telegram"},
    )
    assert len(results) >= 1
    assert all("北京" in r[0] for r in results)
