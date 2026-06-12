"""查询重写器：消解代词 + 意图分类 + 返回改写后的查询。

设计原则：fail-open —— LLM 调用超时或解析失败时，回退使用原始查询，确保不阻断检索链路。
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class GateDecision:
    """门控决策结果"""

    needs_episodic: bool  # 是否需要历史记忆检索
    episodic_query: str  # 消解代词后的查询
    procedure_query: str = ""  # 过程类查询改写
    needs_procedure: bool = False  # 是否需要过程/偏好检索


class QueryRewriter:
    """轻量 LLM 驱动的查询重写器。

    并行执行两路：
    1. 历史感知改写：判断 RETRIEVE/NO_RETRIEVE，消解代词
    2. 过程查询改写：将模糊请求转为结构化摘要模式

    800ms 超时 + fail-open：任一路失败均回退原始查询。
    """

    # 允许的决策值
    _VALID_DECISIONS = {"RETRIEVE", "NO_RETRIEVE"}

    def __init__(
        self,
        llm_client: Any,  # 接受 chat(messages, ...) 接口的 LLM 客户端
        timeout_ms: int = 800,
        max_tokens: int = 220,
    ):
        self._llm = llm_client
        self._timeout_s = max(0.1, float(timeout_ms) / 1000.0)
        self._max_tokens = max(64, int(max_tokens))

    async def decide(self, user_msg: str, recent_history: str = "") -> GateDecision:
        """对用户消息执行门控决策和查询重写。

        Args:
            user_msg: 当前用户消息
            recent_history: 近期对话历史文本

        Returns:
            GateDecision: 如果 LLM 解析失败，needs_episodic=True, episodic_query=user_msg
        """
        fallback = GateDecision(
            needs_episodic=True,
            episodic_query=user_msg,
        )

        try:
            # 并行：历史感知改写 + procedure 改写
            main_task = asyncio.create_task(
                self._call_llm(self._build_prompt(user_msg, recent_history))
            )
            procedure_task = asyncio.create_task(
                self._rewrite_procedure_query(user_msg)
            )

            done, pending = await asyncio.wait(
                {main_task, procedure_task},
                timeout=self._timeout_s,
            )
            for task in pending:
                task.cancel()

            raw_output = ""
            procedure_query = ""

            if main_task in done:
                try:
                    raw_output = main_task.result()
                except Exception:
                    raw_output = ""

            if procedure_task in done:
                try:
                    procedure_query = procedure_task.result()
                except Exception:
                    procedure_query = ""

            # 解析 LLM 输出
            decision = self._parse_output(raw_output)
            if decision is None:
                return fallback

            return GateDecision(
                needs_episodic=decision.get("needs_episodic", True),
                episodic_query=decision.get("episodic_query", user_msg) or user_msg,
                procedure_query=procedure_query,
                needs_procedure=bool(procedure_query and procedure_query.strip()),
            )

        except Exception:
            return fallback

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM，返回原始文本。"""
        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            max_tokens=self._max_tokens,
            disable_thinking=True,
        )
        content = getattr(response, "content", response)
        return str(content or "")

    def _parse_output(self, raw: str) -> dict | None:
        """从 LLM 输出中解析 XML 格式的决策结果。"""
        if not raw or not raw.strip():
            return None

        text = raw.strip()

        # 提取 <decision>
        decision_match = re.search(r"<decision>\s*(RETRIEVE|NO_RETRIEVE)\s*</decision>", text, re.IGNORECASE)
        if not decision_match:
            return None

        decision_val = decision_match.group(1).upper()
        if decision_val not in self._VALID_DECISIONS:
            return None

        # 提取 <history_query>
        query_match = re.search(r"<history_query>\s*(.+?)\s*</history_query>", text, re.DOTALL)
        episodic_query = query_match.group(1).strip() if query_match else ""

        return {
            "needs_episodic": decision_val == "RETRIEVE",
            "episodic_query": episodic_query,
        }

    async def _rewrite_procedure_query(self, user_msg: str) -> str:
        """将用户消息改写为适合过程/偏好匹配的摘要形式。"""
        try:
            prompt = self._build_procedure_prompt(user_msg)
            raw = await self._call_llm(prompt)
            return self._clean_procedure_query(raw)
        except Exception:
            return ""

    def _clean_procedure_query(self, raw: str) -> str:
        """清洗 procedure 查询输出，过滤哨兵值。"""
        text = raw.strip()
        # 压缩空白
        text = re.sub(r"\s+", " ", text)
        # 过滤哨兵值
        sentinels = {"空", "无", "none", "null", "(empty)", "暂无", "没有"}
        if text.lower().rstrip(".") in sentinels:
            return ""
        return text

    def _build_prompt(self, user_msg: str, recent_history: str) -> str:
        """构建历史感知改写的 LLM prompt。"""
        history_block = ""
        if recent_history and recent_history.strip():
            history_block = f"""## 近期对话历史
{recent_history.strip()}

"""

        return f"""你是查询改写代理。根据用户消息和对话历史，判断是否需要检索记忆，并改写查询。

{history_block}## 当前用户消息
{user_msg}

## 判断规则
- 简单问候（"你好""hi"）→ NO_RETRIEVE
- 询问过去的事件、个人事实 → RETRIEVE
- 包含代词（"上次/那个/它"）→ RETRIEVE 并消解为具体实体
- 闲聊、通用知识、简单确认 → NO_RETRIEVE

## 输出格式
<decision>RETRIEVE</decision>
<history_query>消解代词后的具体查询</history_query>

或

<decision>NO_RETRIEVE</decision>
<history_query></history_query>"""

    def _build_procedure_prompt(self, user_msg: str) -> str:
        """构建 procedure 查询改写的 LLM prompt。"""
        return f"""将以下用户消息改写为适合检索操作规范和偏好的摘要形式。

- 去除一次性标题、情绪词
- 保留可复用的类别词
- 改写为第三人称

用户消息：{user_msg}

输出一行摘要文本（不要任何解释）："""
