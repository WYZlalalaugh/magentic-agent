"""HyDE (Hypothetical Document Embeddings) 假设文档生成器。

用轻量 LLM 生成两条假设记忆文档（event-style + general-style），
作为辅助查询与原始查询一起进入向量检索，弥补用户措辞和记忆措辞不匹配的问题。

设计原则：超时优雅降级 —— 假设生成失败时直接跳过，只用原始查询检索。
"""

import asyncio
from typing import Any


class HyDEEnhancer:
    """假设文档生成器。

    每次查询并行生成两条假设：
    - event-style: 时间戳事件风格（"用户在 X 日做了 Y"）
    - general-style: 通用事实风格（"用户偏好/是 X"）

    轻量 LLM (80 tokens, 2s 超时) 确保低延迟低成本。
    """

    def __init__(
        self,
        llm_client: Any,
        timeout_ms: int = 2000,
        max_tokens: int = 80,
    ):
        self._llm = llm_client
        self._timeout_s = max(0.1, float(timeout_ms) / 1000.0)
        self._max_tokens = max(16, int(max_tokens))

    async def generate_hypotheses(self, query: str) -> list[str]:
        """生成假设文档列表。

        Returns:
            假设文本列表（空列表表示生成失败或超时）
        """
        try:
            tasks = [
                self._gen_hypothesis(query, style="event"),
                self._gen_hypothesis(query, style="general"),
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self._timeout_s,
            )
            return [r for r in results if isinstance(r, str) and r.strip()]
        except (asyncio.TimeoutError, Exception):
            return []

    async def _gen_hypothesis(self, query: str, style: str) -> str | None:
        """生成单条假设文档。"""
        try:
            prompt = self._build_prompt(query, style)
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                max_tokens=self._max_tokens,
            )
            content = getattr(response, "content", response)
            text = str(content or "").strip()
            return text if text else None
        except Exception:
            return None

    def _build_prompt(self, query: str, style: str) -> str:
        """构建假设生成的 prompt（参考 magentic-agent 的 HyDE 实现）。"""
        if style == "event":
            return f"""Generate a hypothetical memory entry with a specific time, format like '[2026-03-08] User...'.
Keep the original query's semantic polarity — if the query asks about likes, generate positive; if it asks about dislikes, generate negative.
Write in third-person, one line only.

Query: {query}

Output only the entry:"""
        else:
            return f"""Generate a hypothetical memory entry as if it existed in the database.
Always generate an affirmative, third-person declarative statement ('User...').
Keep it concise and factual. Keep the original query's semantic polarity.
Do not add information not present in the query.

Query: {query}

Output only the entry:"""
