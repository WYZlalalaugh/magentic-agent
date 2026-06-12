"""MarkdownOptimizer — 定时归档任务。

每 18 小时（可配置）执行一次：
1. 原子快照 PENDING.md → PENDING.snapshot.md
2. LLM 合并 PENDING + MEMORY → 新 MEMORY.md
3. LLM 更新 SELF.md（三段自我认知）
4. 提交或回滚
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MarkdownOptimizer:
    """PENDING 缓冲归档优化器。

    将高频增量写入（PENDING.md）和低频全量更新（MEMORY.md）解耦，
    保护 prompt 缓存在多轮对话中稳定复用。
    """

    def __init__(
        self,
        markdown_store: Any,  # MarkdownMemoryStore
        llm_client: Any = None,
        model: str = "",
        interval_hours: float = 18.0,
    ):
        self._store = markdown_store
        self._llm = llm_client
        self._model = model
        self._interval_seconds = max(3600, int(interval_hours * 3600))
        self._lock = asyncio.Lock()

    async def run(self, user_id: str) -> bool:
        """对指定用户执行一次合并归档。

        Returns:
            True 表示成功，False 表示跳过（没有新内容或失败）
        """
        async with self._lock:
            pending = self._store.read_pending(user_id)
            if not pending or not pending.strip():
                logger.info("MarkdownOptimizer: no pending items for user=%s, skip", user_id)
                return False

            # 1. 原子快照
            snapshot = self._snapshot_pending(user_id, pending)
            if not snapshot:
                return False

            try:
                # 2. 合并 MEMORY.md
                current_memory = self._store.read_long_term(user_id)
                merged = await self._merge_memory(current_memory, snapshot, user_id)
                if merged:
                    self._store.write_long_term(user_id, merged)

                # 3. 更新 SELF.md
                current_self = self._store.read_self(user_id)
                updated_self = await self._update_self(current_self, snapshot, user_id)
                if updated_self:
                    self._store.write_self(user_id, updated_self)

                # 4. 提交：清空 PENDING + 删除快照
                self._store.write_pending(user_id, "")
                self._delete_snapshot(user_id)
                logger.info("MarkdownOptimizer: optimized user=%s", user_id)
                return True

            except Exception as e:
                logger.warning("MarkdownOptimizer: failed for user=%s, rollback: %s", user_id, e)
                self._rollback(user_id, pending)
                return False

    def _snapshot_pending(self, user_id: str, content: str) -> str | None:
        """创建 PENDING.snapshot.md 原子快照。"""
        try:
            snapshot_path = self._store._user_dir(user_id) / "PENDING.snapshot.md"
            snapshot_path.write_text(content, encoding="utf-8")
            return content
        except Exception:
            return None

    def _delete_snapshot(self, user_id: str):
        try:
            path = self._store._user_dir(user_id) / "PENDING.snapshot.md"
            if path.exists():
                path.unlink()
        except Exception:
            pass

    def _rollback(self, user_id: str, original_pending: str):
        """失败时回滚——把 snapshot 内容合并回 PENDING。"""
        try:
            snapshot_path = self._store._user_dir(user_id) / "PENDING.snapshot.md"
            snapshot = ""
            if snapshot_path.exists():
                snapshot = snapshot_path.read_text(encoding="utf-8")
                snapshot_path.unlink()
            merged = (original_pending + snapshot).strip()
            self._store.write_pending(user_id, merged)
        except Exception:
            pass

    async def _merge_memory(self, current: str, pending: str, user_id: str) -> str:
        """LLM 合并 PENDING 事实到 MEMORY。"""
        if self._llm is None:
            return ""

        prompt = f"""你是记忆优化代理。将待合并事实合并到当前长期记忆中。

规则：
1. 同一个人/偏好的多条事实合并为一条
2. 移除过时或矛盾的信息（保留最新的）
3. 保留现有记忆中与新事实不冲突的部分
4. 每条事实用 bullet 格式: "- [tag] 内容"
5. tag 仅限于: identity, preference, key_info, health_long_term, requested_memory, correction, agent_context

## 当前长期记忆
{current or "（空）"}

## 待合并事实
{pending}

## 输出
直接输出完整的 MEMORY.md 内容（不要 JSON 包裹，不要解释）："""

        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
            )
            content = getattr(response, "content", response)
            return str(content or "").strip()
        except Exception as e:
            logger.warning("MarkdownOptimizer: merge_memory LLM failed: %s", e)
            return current

    async def _update_self(self, current_self: str, pending: str, user_id: str) -> str:
        """LLM 根据新事实更新 SELF.md。"""
        if self._llm is None:
            return ""

        prompt = f"""你是 Agent 自我认知更新代理。根据用户新事实更新自我认知。

SELF.md 包含三段内容（保持三段结构不变）：
1. Agent 的人格和定位
2. Agent 对当前用户的理解
3. Agent 和当前用户的关系定位

## 当前 SELF.md
{current_self or "（空）"}

## 用户新事实
{pending}

## 输出
直接输出完整的 SELF.md 内容（不要 JSON 包裹，不要解释）："""

        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
            )
            content = getattr(response, "content", response)
            return str(content or "").strip()
        except Exception as e:
            logger.warning("MarkdownOptimizer: update_self LLM failed: %s", e)
            return current_self

    async def run_periodic(self, user_id: str, stop_event: asyncio.Event | None = None):
        """循环执行定时归档。

        在后台 task 中调用此方法。直到 stop_event 被设置。
        """
        while stop_event is None or not stop_event.is_set():
            await self.run(user_id)
            await asyncio.sleep(self._interval_seconds)
