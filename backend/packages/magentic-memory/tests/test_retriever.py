import sys
from pathlib import Path

# Ensure magentic-memory package is importable
_PKG_DIR = str(Path(__file__).parent.parent.resolve())
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from magentic_memory.retriever import rrf_merge


def test_rrf_merge_combines_two_sources():
    """两路都有结果时，RRF 应正确融合"""
    vec_results = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.7}]
    kw_results = [{"id": "c", "score": 0.8}, {"id": "a", "score": 0.5}]
    fused = rrf_merge(vec_results, kw_results, top_n=3, k=60, kw_weight=0.5)
    assert len(fused) == 3
    assert fused[0]["id"] == "a"  # a 在两路都出现，应该排第一
    assert fused[0]["rrf_score"] is not None


def test_rrf_merge_empty_keyword():
    """只有向量结果时直接返回"""
    vec_results = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.5}]
    fused = rrf_merge(vec_results, [], top_n=5)
    assert len(fused) == 2
    assert fused[0]["id"] == "a"


def test_rrf_merge_empty_vector():
    """只有关键词结果时直接返回"""
    kw_results = [{"id": "a", "score": 0.8}, {"id": "b", "score": 0.3}]
    fused = rrf_merge([], kw_results, top_n=5)
    assert len(fused) == 2


def test_rrf_merge_vector_priority():
    """向量权重 1.0 > 关键词权重 0.5，向量命中应排更前"""
    vec_results = [{"id": "a", "score": 0.3}]  # 向量命中
    kw_results = [{"id": "b", "score": 0.9}]  # 关键词命中
    fused = rrf_merge(vec_results, kw_results, top_n=2)
    # a 向量排名=1, 关键词无 → rrf=1/61≈0.0164
    # b 向量无, 关键词排名=1 → rrf=0.5/61≈0.0082
    # 向量通道应排在前面
    assert fused[0]["id"] == "a"


def test_rrf_top_n_truncation():
    """top_n 应正确截断"""
    items = [{"id": str(i), "score": 0.9 - i * 0.1} for i in range(10)]
    fused = rrf_merge(items, [], top_n=3)
    assert len(fused) == 3
    assert fused[0]["id"] == "0"
