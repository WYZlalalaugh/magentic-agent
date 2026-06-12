"""ConsolidationUpdater — 改写 DeerFlow 写入链路。

替代 MemoryUpdater，将 LLM 输出从单一 JSON 改写为：
- history_entries[] → 追加 HISTORY.md + embed Chroma
- pending_items[] → 追加 PENDING.md
- recent_context → 覆盖 RECENT_CONTEXT.md

设计原则：
- 不影响 MemoryMiddleware 的调用方式（复用 MemoryUpdateQueue）
- 和旧 MemoryUpdater 接口兼容
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ConsolidationUpdater:
    """记忆合并更新器：写 Markdown 文件 + Chroma，不写 memory.json。"""

    # PENDING.md 的 7 种合法 tag
    _ALLOWED_TAGS = frozenset({
        "identity", "preference", "key_info",
        "health_long_term", "requested_memory",
        "correction", "agent_context",
    })

    def __init__(
        self,
        markdown_store: Any,  # MarkdownMemoryStore
        vector_store: Any = None,  # VectorMemoryStore | None
        llm_client: Any = None,  # LLM 客户端
        model: str = "",
    ):
        self._store = markdown_store
        self._vector = vector_store
        self._llm = llm_client
        self._model = model

    async def process(
        self,
        messages: list[dict],
        user_id: str,
        source_ref: str = "",
    ) -> dict:
        """处理一轮对话的消息，提取并写入记忆。

        Args:
            messages: 对话消息列表 [{"role": "...", "content": "..."}, ...]
            user_id: 用户 ID
            source_ref: 来源引用

        Returns:
            {"history_entries": [...], "pending_items": [...], "recent_context": "..."}
        """
        if not messages or self._llm is None:
            return {"history_entries": [], "pending_items": [], "recent_context": ""}

        # 1. 调用 LLM 提取结构化信息
        conversation = self._format_conversation(messages)
        current_memory = self._store.read_long_term(user_id)
        current_history = self._store.read_history(user_id)

        prompt = self._build_prompt(conversation, current_memory, current_history)
        raw_output = await self._call_llm(prompt)

        result = self._parse_output(raw_output)

        # 2. 写入 HISTORY.md
        history_entries = result.get("history_entries", [])
        if history_entries:
            history_text = "\n".join(history_entries) + "\n"
            self._store.append_history(user_id, history_text)

            # embed 到 Chroma
            if self._vector:
                for entry in history_entries:
                    try:
                        self._vector.add_memory(
                            memory_type="event",
                            content=entry,
                            metadata={"source_ref": source_ref},
                        )
                    except Exception:
                        logger.debug("consolidation: embed event failed for %r", entry[:60])

        # 3. 写入 PENDING.md
        pending_items = result.get("pending_items", [])
        if pending_items:
            formatted = self._format_pending_items(pending_items)
            if formatted:
                self._store.append_pending(user_id, formatted)

        # 4. 写入 RECENT_CONTEXT.md
        recent_context = result.get("recent_context", "")
        if recent_context:
            self._store.write_recent_context(user_id, recent_context)

        return result

    def _format_conversation(self, messages: list[dict]) -> str:
        lines = []
        for msg in messages:
            role = str(msg.get("role", "") or "").upper()
            content = str(msg.get("content", "") or "").strip()
            if role in ("USER", "ASSISTANT") and content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_prompt(self, conversation: str, current_memory: str, current_history: str) -> str:
        return f"""你是记忆提取代理。从对话中提取结构化信息。

## 当前长期记忆
{current_memory or "（空）"}

## 当前历史
{current_history or "（空）"}

## 对话内容
{conversation}

## 输出格式（JSON）
{{
  "history_entries": ["[YYYY-MM-DD HH:MM] 摘要1", "..."],
  "pending_items": [{{"tag": "identity", "content": "..."}}],
  "recent_context": "# Recent Context\\n..."
}}

pending_items 的 tag 仅限于: identity, preference, key_info, health_long_term, requested_memory, correction, agent_context。

只输出 JSON。"""

    def _format_pending_items(self, items: list[dict]) -> str:
        lines = []
        for item in items:
            tag = str(item.get("tag", "")).strip()
            content = str(item.get("content", "")).strip()
            if tag in self._ALLOWED_TAGS and content:
                lines.append(f"- [{tag}] {content}")
        return "\n".join(lines) + "\n" if lines else ""

    def _parse_output(self, raw: str) -> dict:
        import json

        try:
            text = raw.strip()
            if "```" in text:
                text = text.split("```json")[-1].split("```")[0].strip()
            return json.loads(text)
        except Exception:
            logger.warning("ConsolidationUpdater: failed to parse LLM output")
            return {"history_entries": [], "pending_items": [], "recent_context": ""}

    async def _call_llm(self, prompt: str) -> str:
        if self._llm is None:
            return ""
        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            model=self._model,
        )
        content = getattr(response, "content", response)
        return str(content or "")
