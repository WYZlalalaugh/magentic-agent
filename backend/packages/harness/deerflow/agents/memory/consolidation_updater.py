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
            logger.debug("Consolidation: skipped (no messages or LLM)")
            return {"history_entries": [], "pending_items": [], "recent_context": ""}

        logger.info("Consolidation: processing %d messages for user=%s", len(messages), user_id)

        # 1. 调用 LLM 提取结构化信息
        conversation = self._format_conversation(messages)
        current_memory = self._store.read_long_term(user_id)
        current_history = self._store.read_history(user_id)

        prompt = self._build_prompt(conversation, current_memory, current_history)
        raw_output = await self._call_llm(prompt)
        logger.info("Consolidation: LLM returned:\n%s", (raw_output or "EMPTY")[:1000])

        result = self._parse_output(raw_output)
        logger.info("Consolidation: parsed %d history_entries, %d pending_items, recent_context=%r",
                     len(result.get("history_entries", [])),
                     len(result.get("pending_items", [])),
                     (result.get("recent_context", "") or "")[:60])

        # 2. 写入 HISTORY.md（去重：跳过已存在的条目）
        history_entries = result.get("history_entries", [])
        history_lines = []
        if history_entries:
            current = self._store.read_history(user_id)
            current_events = set(
                line.strip() for line in current.splitlines() if line.strip()
            )
            for entry in history_entries:
                if isinstance(entry, str):
                    if entry.strip() not in current_events:
                        history_lines.append(entry)
                elif isinstance(entry, dict):
                    summary = entry.get("summary", "")
                    if summary and summary.strip() not in current_events:
                        history_lines.append(summary)
        if history_lines:
            history_text = "\n".join(history_lines) + "\n"
            self._store.append_history(user_id, history_text)
            logger.info("Consolidation: wrote %d history entries", len(history_lines))

            # embed 到 Chroma
            if self._vector:
                for line in history_lines:
                    try:
                        self._vector.add_memory(
                            memory_type="event",
                            content=line,
                            metadata={"source_ref": source_ref},
                        )
                    except Exception:
                        logger.debug("consolidation: embed event failed for %r", line[:60])

        # 3. 写入 PENDING.md（去重：跳过内容相同的条目）
        pending_items = result.get("pending_items", [])
        if pending_items:
            current_pending = self._store.read_pending(user_id)
            formatted = self._format_pending_items(pending_items)
            if formatted:
                # 只在内容不重复时才追加
                new_lines = [
                    line for line in formatted.splitlines()
                    if line.strip() and line.strip() not in current_pending
                ]
                if new_lines:
                    self._store.append_pending(user_id, "\n".join(new_lines) + "\n")
                    logger.info("Consolidation: wrote %d pending items", len(new_lines))

        # 4. 写入 RECENT_CONTEXT.md
        recent_context = result.get("recent_context", "")
        if isinstance(recent_context, dict):
            # LLM returned structured object → format as markdown text
            lines = ["# Recent Context", ""]
            for key, label in [
                ("active_topics", "最近持续关注"),
                ("user_preferences", "最近明确偏好"),
                ("follow_ups", "最近待延续话题"),
                ("avoidances", "最近避免事项"),
            ]:
                items = recent_context.get(key, [])
                if items:
                    if isinstance(items, list):
                        lines.append(f"- {label}：{'；'.join(str(i) for i in items[:3])}")
                    else:
                        lines.append(f"- {label}：{items}")
            ongoing = recent_context.get("ongoing_threads", [])
            if ongoing:
                lines.append("")
                lines.append("## Ongoing Threads")
                for t in (ongoing if isinstance(ongoing, list) else [ongoing]):
                    lines.append(f"- {t}")
            recent_context = "\n".join(lines)
        if recent_context:
            self._store.write_recent_context(user_id, str(recent_context))
            logger.info("Consolidation: wrote recent context")

        return result

    def _format_conversation(self, messages: list[dict]) -> str:
        lines = []
        for msg in messages:
            role = str(msg.get("role", "") or "").upper()
            content = str(msg.get("content", "") or "").strip()
            if role in ("USER", "ASSISTANT", "HUMAN", "AI") and content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_prompt(self, conversation: str, current_memory: str, current_history: str) -> str:
        return f"""你是记忆提取代理。从对话中精确提取结构化信息，返回 JSON。

## 字段说明

### 1. "history_entries" → HISTORY.md
按主题拆分，每条 {{"summary":"[YYYY-MM-DD HH:MM] 摘要", "emotional_weight":0}}。
emotional_weight 规则：
- 范围 0-10，默认 0
- 用户明确表达强烈情绪（喜欢/厌恶/受挫/冲突）→ 3-9
- 不确定时保守输出 0

提取规则（严格遵守）：
1. 只提取 USER 明确表达的行动、经历、计划和状态；ASSISTANT 的建议一律不写入
2. 每条必须是简洁的第三人称摘要句，绝对不能包含 "USER:" 或 "ASSISTANT:" 标记
3. 具体细节（名称、地点、数量、价格）必须保留，不得用"某商店""某地方"概括
4. 先判断材料类型：是用户直接自述，还是用户在展示外部聊天记录/transcript
5. 若为 transcript：只写 1 条高层 event；不要猜测 speaker 身份归属
6. transcript 禁止输出未确认关系的句子（"用户向对方透露"等）

### 2. "pending_items" → PENDING.md 候选缓冲
格式：{{"tag": "<tag>", "content": "<string>"}}

tag 限定为 7 个：
- "identity"：稳定背景事实（身份、学校、长期技术方向、经历）
- "preference"：稳定偏好、禁忌、审美、游戏口味
- "key_info"：用户明确允许保存的 key/token/id
- "health_long_term"：长期健康状态（只写长期，不写动态指标）
- "requested_memory"：用户明确要求"长期记住"的内容
- "correction"：对现有记忆的明确纠正
- "agent_context"：工具性配置（端口、环境变量名）

必须遵守：
- 只写跨对话仍有长期价值的内容
- 不写 agent 执行规则、SOP、工具调用顺序
- 不写短期状态、近期计划、日程、一次性操作
- 不写动态健康数据、实时指标

进阶过滤：
- 网络运维细节不提取（内网 IP、路由模式、运营商）
- 临时状态不提取（"最近加班""这周很忙"），规律习惯可提取
- 时效性数字和瞬时情绪不提取，保留背后的价值判断

### 3. "recent_context" → RECENT_CONTEXT.md
输出纯文本字符串，包含 5 个维度的 Markdown bullet 列表：
- 最近持续关注：xxx；xxx
- 最近明确偏好：xxx
- 最近待延续话题：xxx
- 最近避免事项：xxx
- ongoing_threads 严格限制：只有健康、感情、重大生活事件等持续线索才准入；技术讨论、方案脑暴一律不得写入

## 当前长期记忆
{current_memory or "（空）"}

## 当前历史（已有条目，不要重复提取）
{current_history or "（空）"}
{current_history and "以上条目已存在，请只提取本次对话中新增的事件和事实，不要重复旧内容。" or ""}

## 对话内容
{conversation}

## 输出格式（严格按此 JSON 结构，只替换占位内容）
```json
{{
  "history_entries": [
    {{"summary": "[YYYY-MM-DD HH:MM] 用户做了某事", "emotional_weight": 0}}
  ],
  "pending_items": [
    {{"tag": "identity", "content": "用户是..."}}
  ],
  "recent_context": "- 最近持续关注：xxx；xxx\\n- 最近明确偏好：xxx\\n- 最近待延续话题：xxx\\n- 最近避免事项：xxx"
}}
```
history_entries 可选空数组 []，pending_items 可选空数组 []，recent_context 可选空字符串 ""。

只输出 JSON："""

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
        import asyncio
        from langchain_core.messages import HumanMessage
        result = await asyncio.to_thread(self._llm.invoke, [HumanMessage(content=prompt)])
        return str(result.content or "")
