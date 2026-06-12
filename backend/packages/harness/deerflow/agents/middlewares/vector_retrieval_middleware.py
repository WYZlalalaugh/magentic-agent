"""VectorRetrievalMiddleware — 在每条消息到达时执行记忆语义检索并注入结果。

插入位置：DynamicContextMiddleware 之后，SkillActivationMiddleware 之前。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger(__name__)


class VectorRetrievalMiddleware(AgentMiddleware):
    """语义检索中间件。

    在 before_agent 阶段：
    1. 获取当前用户消息和对话历史
    2. 调用 MemoryRetriever 执行完整检索管线（QueryRewriter → HyDE → RRF）
    3. 将检索结果注入到第一条 HumanMessage 的 content 中
    """

    def __init__(
        self,
        retriever: Any = None,  # MemoryRetriever 实例
        enabled: bool = True,
    ):
        self._retriever = retriever
        self._enabled = enabled

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        """在 Agent 推理前执行语义检索并注入结果。"""
        if not self._enabled or self._retriever is None:
            return

        try:
            # 1. 获取当前用户消息
            messages = getattr(state, "messages", [])
            if not messages:
                return

            # 最后一条 HumanMessage 就是当前用户消息
            user_msg = ""
            history = []
            for msg in reversed(messages):
                role = _get_role(msg)
                if role == "user" and not user_msg:
                    user_msg = _get_content(msg)
                    continue
                if role in ("user", "assistant") and user_msg:
                    history.insert(0, f"[{role}] {_get_content(msg)[:200]}")

            if not user_msg:
                return

            # 2. 执行检索
            history_text = "\n".join(history[-10:])  # 最近 10 轮
            results = await self._retriever.retrieve(
                query=user_msg,
                history=history_text,
            )

            if not results:
                return

            # 3. 格式化注入内容
            injection = self._format_injection(results)

            # 4. 注入到第一条消息（通常是 system-reminder 所在的消息）
            # 或注入到最旧的 HumanMessage
            target_msg = self._find_injection_target(messages)
            if target_msg is not None:
                existing = _get_content(target_msg)
                _set_content(target_msg, existing + injection)

        except Exception:
            logger.exception("VectorRetrievalMiddleware: retrieval failed, skipping")

    def _format_injection(self, results: list[dict]) -> str:
        """格式化检索结果为注入文本。"""
        parts = ["\n\n<semantic_memory>", "以下记忆条目来自系统语义检索，不是用户陈述：", ""]

        # 按类型分组：procedure > preference > event > profile
        by_type: dict[str, list[dict]] = {}
        for r in results[:12]:  # 最多 12 条
            mtype = str(r.get("memory_type", "event"))
            by_type.setdefault(mtype, []).append(r)

        for mtype in ("procedure", "preference", "event", "profile"):
            items = by_type.get(mtype, [])
            if not items:
                continue
            label = {"procedure": "操作规范", "preference": "用户偏好", "event": "相关历史", "profile": "用户画像"}.get(mtype, mtype)
            parts.append(f"【{label}】")
            for item in items:
                content = str(item.get("content", "")).strip()
                score = item.get("rrf_score") or item.get("score", 0)
                if content:
                    confidence = ""
                    if float(score) < 0.5:
                        confidence = " | 有印象，不确定"
                    parts.append(f"- {content}{confidence}")
            parts.append("")

        parts.append("</semantic_memory>")
        return "\n".join(parts)

    def _find_injection_target(self, messages: list) -> Any | None:
        """找到合适的目标消息用于注入检索结果。

        优先注入到第一条 HumanMessage；如果没有，注入到最后一条。
        """
        for msg in messages:
            if _get_role(msg) == "user":
                return msg
        return messages[-1] if messages else None


def _get_role(msg: Any) -> str:
    """从 LangChain 消息对象提取 role。"""
    if isinstance(msg, dict):
        return str(msg.get("role", "")).lower()
    role = getattr(msg, "type", None)
    if role is None:
        role = getattr(msg, "role", None)
    return str(role or "").lower()


def _get_content(msg: Any) -> str:
    """从 LangChain 消息对象提取 content。"""
    if isinstance(msg, dict):
        return str(msg.get("content", "") or "")
    return str(getattr(msg, "content", "") or "")


def _set_content(msg: Any, content: str) -> None:
    """设置 LangChain 消息的 content。"""
    if isinstance(msg, dict):
        msg["content"] = content
    elif hasattr(msg, "content"):
        msg.content = content
