from pathlib import Path


class MarkdownMemoryStore:
    """用五个 Markdown 文件存储用户记忆。

    文件结构:
        {base_dir}/users/{user_id}/memory/
        ├── MEMORY.md          # 用户画像全文（注入 prompt）
        ├── HISTORY.md         # 时间线事件（embed 到 Chroma）
        ├── SELF.md            # Agent 自我认知
        ├── RECENT_CONTEXT.md  # 当前语境压缩
        └── PENDING.md         # 增量写入缓冲
    """

    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)

    def _user_dir(self, user_id: str) -> Path:
        return self._base_dir / "users" / user_id / "memory"

    def _ensure_dir(self, user_id: str) -> Path:
        d = self._user_dir(user_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _read_file(self, user_id: str, filename: str) -> str:
        path = self._user_dir(user_id) / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _write_file(self, user_id: str, filename: str, content: str):
        self._ensure_dir(user_id)
        (self._user_dir(user_id) / filename).write_text(content, encoding="utf-8")

    def _append_file(self, user_id: str, filename: str, content: str):
        self._ensure_dir(user_id)
        with open(self._user_dir(user_id) / filename, "a", encoding="utf-8") as f:
            f.write(content)

    # ── MEMORY.md ──

    def read_long_term(self, user_id: str) -> str:
        return self._read_file(user_id, "MEMORY.md")

    def write_long_term(self, user_id: str, content: str):
        self._write_file(user_id, "MEMORY.md", content)

    # ── HISTORY.md ──

    def read_history(self, user_id: str) -> str:
        return self._read_file(user_id, "HISTORY.md")

    def write_history(self, user_id: str, content: str):
        self._write_file(user_id, "HISTORY.md", content)

    def append_history(self, user_id: str, content: str):
        self._append_file(user_id, "HISTORY.md", content)

    # ── SELF.md ──

    def read_self(self, user_id: str) -> str:
        return self._read_file(user_id, "SELF.md")

    def write_self(self, user_id: str, content: str):
        self._write_file(user_id, "SELF.md", content)

    # ── RECENT_CONTEXT.md ──

    def read_recent_context(self, user_id: str) -> str:
        return self._read_file(user_id, "RECENT_CONTEXT.md")

    def write_recent_context(self, user_id: str, content: str):
        self._write_file(user_id, "RECENT_CONTEXT.md", content)

    # ── PENDING.md ──

    def read_pending(self, user_id: str) -> str:
        return self._read_file(user_id, "PENDING.md")

    def append_pending(self, user_id: str, content: str):
        self._append_file(user_id, "PENDING.md", content)

    def write_pending(self, user_id: str, content: str):
        self._write_file(user_id, "PENDING.md", content)
