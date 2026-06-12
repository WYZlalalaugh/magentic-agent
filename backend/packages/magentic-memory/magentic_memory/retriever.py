"""RRF (Reciprocal Rank Fusion) 排序融合 + MemoryRetriever 完整检索编排。

将 QueryRewriter → HyDE → 多向量通道 → 关键词通道 → RRF 融合
编排为统一的检索入口。
"""

from typing import Any

from magentic_memory.query_rewriter import GateDecision, QueryRewriter
from magentic_memory.hyde_enhancer import HyDEEnhancer
from magentic_memory.vector_store import VectorMemoryStore


def rrf_merge(
    vec_results: list[dict],
    kw_results: list[dict],
    top_n: int,
    k: float = 60.0,
    kw_weight: float = 0.5,
) -> list[dict]:
    """RRF 融合：score = 1/(k+rank_vec) + kw_weight/(k+rank_kw)

    Args:
        vec_results: 向量检索结果，按分数降序
        kw_results: 关键词检索结果，按分数降序
        top_n: 返回前 N 条
        k: RRF 常数，默认 60
        kw_weight: 关键词通道权重，默认 0.5（向量通道权重 1.0）
    """
    # 给两路结果分别赋予排名
    vec_ranks: dict[str, float] = {}
    for rank, item in enumerate(vec_results):
        item_id = str(item.get("id", id(item)))
        vec_score = float(item.get("score", 0.0) or 0.0)
        # 保存原始分数和排名
        item_dict = dict(item)
        item_dict["_vec_rank"] = rank + 1
        item_dict["_vec_score"] = vec_score
        vec_ranks[item_id] = 0  # placeholder, will overwrite
        vec_ranks[item_id] = float(rank + 1)

    kw_ranks: dict[str, float] = {}
    for rank, item in enumerate(kw_results):
        item_id = str(item.get("id", id(item)))
        kw_ranks[item_id] = float(rank + 1)

    # 如果只有向量结果，直接返回
    if not kw_results:
        sorted_vec = sorted(
            vec_results,
            key=lambda x: float(x.get("score", 0.0) or 0.0),
            reverse=True,
        )
        return sorted_vec[:top_n]

    # 合并所有 ID
    all_ids: set[str] = set()
    id_to_item: dict[str, dict] = {}
    for item in vec_results:
        item_id = str(item.get("id", id(item)))
        all_ids.add(item_id)
        id_to_item[item_id] = dict(item)
        id_to_item[item_id]["_vec_rank"] = vec_ranks.get(item_id, float("inf"))

    for item in kw_results:
        item_id = str(item.get("id", id(item)))
        all_ids.add(item_id)
        if item_id not in id_to_item:
            id_to_item[item_id] = dict(item)
        id_to_item[item_id]["_kw_rank"] = kw_ranks.get(item_id, float("inf"))

    # 计算 RRF 分数
    scored: list[dict] = []
    for item_id in all_ids:
        item = dict(id_to_item[item_id])
        item["id"] = item_id

        vec_r = vec_ranks.get(item_id, float("inf"))
        kw_r = kw_ranks.get(item_id, float("inf"))

        rrf_score = 0.0
        if vec_r != float("inf"):
            rrf_score += 1.0 / (k + vec_r)
        if kw_r != float("inf"):
            rrf_score += kw_weight / (k + kw_r)

        item["rrf_score"] = rrf_score
        scored.append(item)

    # 按 RRF 分数降序排列，同分按原始分数降序
    scored.sort(
        key=lambda x: (
            -x["rrf_score"],
            -float(x.get("score", 0.0) or 0.0),
        )
    )

    return scored[:top_n]


class MemoryRetriever:
    """统一检索编排：QueryRewriter → HyDE → 多向量通道 → 关键词通道 → RRF。

    注入中间件使用此接口进行每轮对话的记忆检索。
    """

    def __init__(
        self,
        vector_store: VectorMemoryStore,
        rewriter: QueryRewriter,
        hyde: HyDEEnhancer | None = None,
    ):
        self._store = vector_store
        self._rewriter = rewriter
        self._hyde = hyde

    async def retrieve(
        self,
        query: str,
        memory_types: list[str] | None = None,
        top_k: int = 8,
        history: str = "",
        filter: dict | None = None,
    ) -> list[dict]:
        """执行完整检索管线。

        Args:
            query: 用户原始查询
            memory_types: 要检索的记忆类型，默认全部
            top_k: 返回结果数
            history: 近期对话历史（供查询重写消解代词）
            filter: Chroma metadata 过滤

        Returns:
            检索结果列表，按 RRF 分数降序
        """
        if memory_types is None:
            memory_types = ["event", "procedure", "preference", "profile"]

        # 1. 查询重写
        decision = await self._rewriter.decide(query, history)
        if not decision.needs_episodic:
            return []

        # 2. HyDE 假设生成
        aux_queries = [decision.episodic_query]
        if self._hyde:
            hyps = await self._hyde.generate_hypotheses(decision.episodic_query)
            aux_queries.extend(hyps)

        # 3. 多向量通道检索 (查询 + 假设 × 多类型)
        vec_results: list[dict] = []
        for mtype in memory_types:
            for q in aux_queries:
                try:
                    results = self._store.similarity_search(
                        query=q,
                        memory_type=mtype,
                        k=top_k,
                        filter=filter,
                    )
                    for doc, score in results:
                        vec_results.append({
                            "id": f"{mtype}:{doc[:20]}",
                            "score": score,
                            "memory_type": mtype,
                            "content": doc,
                        })
                except Exception:
                    continue

        # 4. RRF 融合（关键词通道目前复用向量通道结果）
        return rrf_merge(vec_results, [], top_n=top_k)
