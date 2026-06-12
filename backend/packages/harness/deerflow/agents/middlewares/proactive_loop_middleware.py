"""ProactiveLoopMiddleware — 集成 ProactiveLoop 到 DeerFlow Gateway。

在 Gateway 启动时创建独立的主动推送循环，不阻塞被动回复链路。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ProactiveLoopMiddleware:
    """Gateway 级别的主动推送中间件。

    在 Gateway 启动时调用 start()，创建后台 asyncio task。
    在 Gateway 关闭时调用 stop()，优雅退出。
    """

    def __init__(
        self,
        *,
        # ── 从 DeerFlow 环境注入 ──
        mcp_pool: Any = None,           # McpClientPool 实例
        llm_client: Any = None,          # LLM provider（与被动回复共用）
        llm_model: str = "",             # LLM 模型名
        channel_manager: Any = None,     # ChannelManager 实例
        proactive_sources_path: str = "",  # proactive_sources.json 路径

        # ── 从配置注入 ──
        enabled: bool = True,
        default_channel: str = "telegram",
        tick_s3: int = 60,
        tick_s2: int = 120,
        tick_s1: int = 240,
        tick_s0: int = 480,
        score_weight_energy: float = 0.35,
        jitter: float = 0.3,
        drift_enabled: bool = True,
        drift_skills_dir: str = "",
        drift_min_interval_hours: float = 3.0,
    ):
        self._enabled = enabled
        self._mcp_pool = mcp_pool
        self._llm = llm_client
        self._llm_model = llm_model
        self._channel_manager = channel_manager
        self._sources_path = proactive_sources_path
        self._default_channel = default_channel

        self._tick_s3 = tick_s3
        self._tick_s2 = tick_s2
        self._tick_s1 = tick_s1
        self._tick_s0 = tick_s0
        self._score_weight = score_weight_energy
        self._jitter = jitter
        self._drift_enabled = drift_enabled
        self._drift_skills_dir = drift_skills_dir
        self._drift_cooldown = drift_min_interval_hours

        self._loop: Any = None
        self._task: asyncio.Task | None = None

    async def start(self):
        """启动主动推送循环（作为后台 asyncio task）。"""
        if not self._enabled:
            logger.info("ProactiveLoopMiddleware: disabled, skipping")
            return

        # 1. 构建推送函数：复用 DeerFlow ChannelManager 发送消息
        async def push_fn(content: str):
            if self._channel_manager is None:
                logger.warning("ProactiveLoop: no channel_manager, cannot push")
                return
            try:
                await self._channel_manager.send_message(
                    channel=self._default_channel,
                    chat_id="",  # 需要在 tick 时动态获取
                    text=content,
                )
                logger.info("ProactiveLoop: pushed message")
            except Exception as e:
                logger.exception("ProactiveLoop: push failed: %s", e)

        # 2. 构建数据拉取函数：复用 DeerFlow MCP 连接池
        async def fetch_fn() -> list[dict]:
            if self._mcp_pool is None:
                return []
            try:
                from magentic_proactive_mcp import fetch_content_events_async
                return await fetch_content_events_async(self._mcp_pool)
            except ImportError:
                pass  # MCP source 模块未安装时跳过
            except Exception:
                logger.exception("ProactiveLoop: fetch failed")
            return []

        # 3. 构建 Judge
        from magentic_proactive.judge import ContentJudge
        judge = ContentJudge(
            llm_client=self._llm,
            model=self._llm_model,
        )

        # 4. 构建 Drift 引擎
        drift = None
        if self._drift_enabled and self._drift_skills_dir:
            from magentic_proactive.drift import DriftEngine
            drift = DriftEngine(
                skills_dir=self._drift_skills_dir,
                llm_client=self._llm,
                model=self._llm_model,
            )

        # 5. 构建 ProactiveLoop
        from magentic_proactive.loop import ProactiveLoop
        import magentic_proactive.energy as energy_module

        self._loop = ProactiveLoop(
            judge=judge,
            energy_module=energy_module,
            data_fetcher=fetch_fn,
            push_fn=push_fn,
            drift_engine=drift,
            tick_s3=self._tick_s3,
            tick_s2=self._tick_s2,
            tick_s1=self._tick_s1,
            tick_s0=self._tick_s0,
            score_weight_energy=self._score_weight,
            jitter=self._jitter,
            drift_min_interval_hours=self._drift_cooldown,
        )

        # 6. 启动后台循环
        self._task = asyncio.create_task(self._loop.run())
        logger.info("ProactiveLoopMiddleware: started background task")

    async def stop(self):
        """停止主动推送循环。"""
        if self._loop:
            self._loop.stop()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ProactiveLoopMiddleware: stopped")

    def record_user_activity(self, channel: str, chat_id: str):
        """记录用户活动时间（由被动回复链路调用）。"""
        if self._loop:
            from datetime import datetime, timezone
            self._loop.record_user_activity(datetime.now(timezone.utc))
