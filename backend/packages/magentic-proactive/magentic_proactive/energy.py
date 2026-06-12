"""电量模型：多时间尺度指数衰减 + 自适应轮询间隔计算。

E(t) = α·exp(-t/τ₁) + β·exp(-t/τ₂) + γ·exp(-t/τ₃)

三个时间尺度模拟用户活跃度衰减：
- 短期 (30min)：对话余温
- 中期 (4h)：同日语境
- 长期 (48h)：关系连续性
"""

import math
import random


def compute_energy(minutes_since_last: float) -> float:
    """计算电量值 [0, 1]。

    0 = 用户已完全离线（>72h）
    1 = 用户刚发完消息（0min）

    Args:
        minutes_since_last: 距离用户上次消息的分钟数
    """
    alpha, beta, gamma = 0.50, 0.35, 0.15
    tau1, tau2, tau3 = 30.0, 240.0, 2880.0

    result = (
        alpha * math.exp(-minutes_since_last / tau1)
        + beta * math.exp(-minutes_since_last / tau2)
        + gamma * math.exp(-minutes_since_last / tau3)
    )
    return max(0.0, min(1.0, result))


def d_energy(energy: float) -> float:
    """互动饥渴度 = 1 - 电量。

    电量越低 → 越久没互动 → 饥渴度越高 → 应更频繁轮询。
    """
    return 1.0 - max(0.0, min(1.0, energy))


def next_tick_from_score(
    base_score: float,
    tick_s3: int,
    tick_s2: int,
    tick_s1: int,
    tick_s0: int,
    jitter: float = 0.3,
) -> int:
    """根据紧迫性分数计算下一次轮询间隔（秒）。

    四档阈值：
        base_score > 0.70  → s3 (最短间隔，最高紧迫性)
        base_score > 0.40  → s2
        base_score > 0.20  → s1
        else               → s0 (最长间隔，最低紧迫性)

    加 ±jitter 随机抖动防止可预测行为。

    Args:
        base_score: [0, 1] 紧迫性分数
        tick_s3...s0: 四档间隔值（秒），必须满足 s0 ≥ s1 ≥ s2 ≥ s3
        jitter: 抖动幅度，0 = 无抖动，0.3 = ±30%

    Returns:
        轮询间隔，最小 1 秒
    """
    if base_score > 0.70:
        base = tick_s3
    elif base_score > 0.40:
        base = tick_s2
    elif base_score > 0.20:
        base = tick_s1
    else:
        base = tick_s0

    jittered = base * (1.0 + random.uniform(-jitter, jitter))
    return max(1, int(jittered))


def get_polling_interval(
    minutes_since_last: float,
    score_weight: float = 0.35,
    tick_s3: int = 60,
    tick_s2: int = 120,
    tick_s1: int = 240,
    tick_s0: int = 480,
    jitter: float = 0.3,
) -> int:
    """一站式接口：从距离上次消息时间计算应等待的下次轮询间隔。

    Args:
        minutes_since_last: 距离用户上次消息的分钟数
        score_weight: 饥渴度权重（默认 0.35）
        tick_s3...s0: 四档间隔（秒）
        jitter: 抖动幅度

    Returns:
        下次轮询间隔（秒）
    """
    energy = compute_energy(minutes_since_last)
    base_score = d_energy(energy) * score_weight
    return next_tick_from_score(base_score, tick_s3, tick_s2, tick_s1, tick_s0, jitter)
