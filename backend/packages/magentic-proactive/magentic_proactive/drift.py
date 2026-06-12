"""Drift 空闲引擎 — 主动推送无内容时执行 SKILL.md 后台任务。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DriftEngine:
    """空闲任务执行引擎。

    扫描 drift/skills/*/SKILL.md 文件，
    在主动推送空闲时选择一个执行。
    """

    def __init__(
        self,
        skills_dir: Path | str,
        llm_client: Any = None,
        model: str = "",
        max_steps: int = 20,
    ):
        self._skills_dir = Path(skills_dir)
        self._llm = llm_client
        self._model = model
        self._max_steps = max_steps

    def scan_skills(self) -> list[dict]:
        """扫描所有可用技能。"""
        skills = []
        if not self._skills_dir.is_dir():
            return skills

        for skill_dir in self._skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue

            skill = self._parse_skill(skill_file)
            if skill:
                skill["dir"] = str(skill_dir)
                skills.append(skill)

        return skills

    def _parse_skill(self, path: Path) -> dict | None:
        """解析 SKILL.md 文件（支持 YAML frontmatter）。"""
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None

        name = path.parent.name
        description = ""
        content = text

        # 简单解析 YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                content = parts[2]
                for line in frontmatter.strip().splitlines():
                    line = line.strip()
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip()
                    elif line.startswith("description:"):
                        description = line.split(":", 1)[1].strip()

        return {
            "name": name,
            "description": description,
            "content": content.strip(),
        }

    async def execute(self) -> dict:
        """执行一次空闲任务。

        Returns:
            {"skill": 技能名, "summary": 执行摘要, "next_action": 后续建议}
        """
        skills = self.scan_skills()
        if not skills:
            logger.debug("DriftEngine: no skills available")
            return {"skill": None, "summary": "无可用技能", "next_action": None}

        # 随机选一个技能执行
        skill = skills[0] if len(skills) == 1 else skills[__import__("random").randint(0, len(skills) - 1)]
        logger.info("DriftEngine: executing skill=%s", skill["name"])

        if self._llm is None:
            return {"skill": skill["name"], "summary": "LLM 未配置，无法执行", "next_action": None}

        prompt = f"""你是后台任务执行代理。按以下技能定义执行任务。

## 技能: {skill['name']}
{skill['description']}

## 任务说明
{skill['content']}

## 要求
1. 按步骤说明逐个执行
2. 每完成一步，输出该步的摘要
3. 全部完成后，输出总结报告
4. 最多执行 {self._max_steps} 步

开始执行（直接输出，不要 JSON 包裹）："""

        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
            )
            content = getattr(response, "content", response)

            return {
                "skill": skill["name"],
                "summary": str(content or "")[:500],
                "next_action": None,
            }
        except Exception as e:
            logger.warning("DriftEngine: LLM call failed: %s", e)
            return {"skill": skill["name"], "summary": f"执行失败: {e}", "next_action": None}
