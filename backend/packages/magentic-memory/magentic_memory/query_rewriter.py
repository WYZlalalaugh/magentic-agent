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
        """构建历史感知改写的 LLM prompt（参考 magentic-agent 验证过的规则）。"""
        history_block = recent_history.strip() or "（无）"
        return f"""你是记忆检索决策器。根据近期对话和当前用户消息，判断是否需要检索用户个人事实或历史事件，并输出查询。

近期对话：
{history_block}

当前用户消息：
{user_msg}

规则：
- NO_RETRIEVE：打招呼、闲聊、确认当前轮内容、通用知识问答、简单回应"好/嗯/继续"、用户提出新的服务偏好或执行规则
- RETRIEVE：询问过去发生的事、用户个人信息、用户是否告诉过某事
- 用户提出新的偏好或规则时，decision 仍是 NO_RETRIEVE；这类内容只交给 procedure/preference 检索处理
- 出现"都有哪些/列举/所有/一共/总共/历史上"这类聚合问题 → RETRIEVE，并改写成覆盖主题的宽泛 query
- 出现"他/她/它/这个/那个/这东西/上次"这类指示词时，优先用近期对话消解为实际实体
- "你还记得吗/你知道我的/你记不记得/我跟你说过"等元问题是在查事实本身，history_query 要贴近记忆摘要
- 提到快递、物流、包裹时，若语境指向用户最近购买行为，应查购买历史
- 提到身体症状、药、复查时，若语境指向用户健康状态，应查健康档案

示例：
用户消息：以后讲复杂问题先给我一个能贯穿始终的例子
→ <decision>NO_RETRIEVE</decision>
   <history_query></history_query>

用户消息：【视频标题】https://short.example/item
→ <decision>NO_RETRIEVE</decision>
   <history_query></history_query>

用户消息：你还记得我用的是哪个 Fitbit 吗
→ <decision>RETRIEVE</decision>
   <history_query>用户使用的 Fitbit 设备型号</history_query>

只输出 XML，不要解释：
<decision>RETRIEVE|NO_RETRIEVE</decision>
<history_query>...</history_query>"""

    def _build_procedure_prompt(self, user_msg: str) -> str:
        """构建 procedure 查询改写的 LLM prompt。"""
        return f"""将以下用户消息改写为适合检索操作规范和偏好的摘要形式。

- 去除一次性标题、情绪词
- 保留可复用的类别词
- 改写为第三人称

用户消息：{user_msg}

输出一行摘要文本（不要任何解释）："""
