"""LLM Agent 分类决策器。

LLM 以 Agent 身份运行，调用工具对内容逐条分类：
- mark_interesting(item_id, reason) → 感兴趣
- mark_not_interesting(item_id) → 不感兴趣
- message_push(draft) → 拟稿推送
- finish_turn(decision) → 完成决策
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContentJudge:
    """LLM Agent 驱动的内容分类器。"""

    def __init__(
        self,
        llm_client: Any,
        model: str = "",
        max_tokens: int = 2048,
    ):
        self._llm = llm_client
        self._model = model
        self._max_tokens = max_tokens

    async def classify_and_decide(
        self,
        items: list[dict],
        user_context: str = "",
        recent_history: str = "",
    ) -> dict:
        """分类内容条目并决定是否推送。

        Args:
            items: 待分类内容 [{"id": "...", "title": "...", "body": "..."}, ...]
            user_context: 用户画像、偏好等上下文
            recent_history: 近期对话历史

        Returns:
            {
                "decision": "reply" | "skip",
                "interesting": [{"id": "...", "reason": "..."}],
                "draft": "推送草稿文本" | None,
            }
        """
        if not items:
            return {"decision": "skip", "interesting": [], "draft": None}

        # 构建 prompt
        prompt = self._build_prompt(items, user_context, recent_history)

        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=self._build_tools(),
                model=self._model,
                max_tokens=self._max_tokens,
            )

            return self._parse_response(response)

        except Exception as e:
            logger.warning("ContentJudge: LLM call failed: %s", e)
            return {"decision": "skip", "interesting": [], "draft": None}

    def _build_prompt(self, items: list[dict], user_context: str, recent_history: str) -> str:
        items_text = ""
        for i, item in enumerate(items):
            items_text += f"\n### 条目 {i + 1}: {item.get('title', '无标题')}\n"
            items_text += f"ID: {item.get('id', 'unknown')}\n"
            items_text += f"内容: {item.get('body', item.get('summary', ''))}\n"

        return f"""你是内容筛选代理。判断以下内容是否值得推送给用户。

## 用户上下文
{user_context or "（无）"}

## 近期对话
{recent_history or "（无）"}

## 待分类内容
{items_text}

## 任务
1. 逐条审视每个条目
2. 调用 mark_interesting 或 mark_not_interesting 分类
3. 如果有感兴趣条目且值得推送，调用 finish_turn(decision="reply")
4. 如果没内容值得推送，调用 finish_turn(decision="skip")"""

    def _build_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "mark_interesting",
                    "description": "标记内容为感兴趣，值得推送",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["item_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_not_interesting",
                    "description": "标记内容为不感兴趣，跳过",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                        },
                        "required": ["item_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_turn",
                    "description": "完成分类决策",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string", "enum": ["reply", "skip"]},
                        },
                        "required": ["decision"],
                    },
                },
            },
        ]

    def _parse_response(self, response: Any) -> dict:
        result = {"decision": "skip", "interesting": [], "draft": None}

        tool_calls = getattr(response, "tool_calls", []) or []
        content = getattr(response, "content", "") or ""

        for tc in tool_calls:
            name = str(getattr(tc, "name", "") or tc.get("name", ""))
            args = getattr(tc, "arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if name == "mark_interesting":
                result["interesting"].append({
                    "id": args.get("item_id", ""),
                    "reason": args.get("reason", ""),
                })
            elif name == "finish_turn":
                result["decision"] = args.get("decision", "skip")

        # 作为草稿，用 content 作为推送文本
        if result["decision"] == "reply" and content:
            result["draft"] = str(content).strip()

        return result
