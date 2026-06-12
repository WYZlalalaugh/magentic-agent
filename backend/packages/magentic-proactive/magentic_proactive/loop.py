"""ProactiveLoop — 独立于被动回复的自适应主动推送循环。

不阻塞 LangGraph 被动回复链路，作为独立的 asyncio 循环运行。
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


class ProactiveLoop:
    """主动推送主循环。

    每轮 tick：
    1. 电量模型 → 计算等待间隔
    2. 拉取数据 → MCP 三路数据源
    3. LLM 分类 → 决策推送或跳过
    4. 空闲 → Drift 后台任务
    """

    def __init__(
        self,
        judge: Any = None,  # ContentJudge
        energy_module: Any = None,  # energy 模块
        data_fetcher: Any = None,  # 数据拉取函数
        push_fn: Any = None,  # 推送函数
        drift_engine: Any = None,  # DriftEngine | None
        *,
        tick_s3: int = 60,
        tick_s2: int = 120,
        tick_s1: int = 240,
        tick_s0: int = 480,
        score_weight_energy: float = 0.35,
        jitter: float = 0.3,
        drift_min_interval_hours: float = 3.0,
    ):
        self._judge = judge
        self._energy = energy_module
        self._fetcher = data_fetcher
        self._push = push_fn
        self._drift = drift_engine

        self._tick_s3 = tick_s3
        self._tick_s2 = tick_s2
        self._tick_s1 = tick_s1
        self._tick_s0 = tick_s0
        self._score_weight = score_weight_energy
        self._jitter = jitter
        self._drift_cooldown = timedelta(hours=drift_min_interval_hours)

        self._running = False
        self._last_user_at: datetime | None = None
        self._last_drift_at: datetime | None = None

    def record_user_activity(self, timestamp: datetime | None = None):
        """记录用户活动时间。"""
        self._last_user_at = timestamp or datetime.now(timezone.utc)

    async def run(self, stop_event: asyncio.Event | None = None):
        """启动主动推送循环。"""
        self._running = True
        logger.info("ProactiveLoop: started")

        try:
            while self._running and (stop_event is None or not stop_event.is_set()):
                interval = self._compute_interval()
                logger.debug("ProactiveLoop: waiting %ds", interval)
                await asyncio.sleep(interval)

                try:
                    await self._tick()
                except Exception:
                    logger.exception("ProactiveLoop: tick failed")
        finally:
            logger.info("ProactiveLoop: stopped")

    def _compute_interval(self) -> int:
        """根据距上次用户活动的时间计算下次轮询间隔。"""
        if self._energy is None or self._last_user_at is None:
            return self._tick_s0

        minutes_since = (datetime.now(timezone.utc) - self._last_user_at).total_seconds() / 60.0
        energy = self._energy.compute_energy(minutes_since)
        d = self._energy.d_energy(energy)
        base_score = d * self._score_weight

        return self._energy.next_tick_from_score(
            base_score,
            tick_s3=self._tick_s3,
            tick_s2=self._tick_s2,
            tick_s1=self._tick_s1,
            tick_s0=self._tick_s0,
            jitter=self._jitter,
        )

    async def _tick(self):
        """执行一轮 tick。"""
        # 1. 拉取数据
        items = []
        if self._fetcher:
            try:
                items = await self._fetcher()
                items = [i for i in (items or []) if isinstance(i, dict)]
            except Exception:
                logger.exception("ProactiveLoop: fetch failed")

        # 2. 有内容 → 分类决策
        if items and self._judge:
            result = await self._judge.classify_and_decide(items)
            if result.get("decision") == "reply" and result.get("draft") and self._push:
                await self._push(result["draft"])
            return

        # 3. 无内容 → Drift
        if self._drift and self._can_drift():
            try:
                self._last_drift_at = datetime.now(timezone.utc)
                await self._drift.execute()
            except Exception:
                logger.exception("ProactiveLoop: drift failed")

    def _can_drift(self) -> bool:
        if self._last_drift_at is None:
            return True
        return (datetime.now(timezone.utc) - self._last_drift_at) >= self._drift_cooldown

    def stop(self):
        self._running = False
