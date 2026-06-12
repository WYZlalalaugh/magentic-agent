"""RRF (Reciprocal Rank Fusion) 排序融合算法。

将向量检索和关键词检索两路结果按倒数排名加权合并。
"""


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
