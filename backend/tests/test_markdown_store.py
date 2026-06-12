import importlib.util
import sys
import tempfile
from pathlib import Path

# 直接加载模块文件，避开 deerflow 包的 langchain 依赖
_STORE_PATH = Path(__file__).parent.parent / "packages" / "harness" / "deerflow" / "agents" / "memory" / "markdown_store.py"
_spec = importlib.util.spec_from_file_location("markdown_store", _STORE_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules["markdown_store"] = _module
_spec.loader.exec_module(_module)
MarkdownMemoryStore = _module.MarkdownMemoryStore


def test_write_and_read_long_term():

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MarkdownMemoryStore(base_dir=Path(tmpdir))
        user_id = "test_user"

        result = store.read_long_term(user_id)
        assert result == ""

        content = "- [identity] 测试用户"
        store.write_long_term(user_id, content)
        assert store.read_long_term(user_id) == content


def test_write_and_read_pending():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MarkdownMemoryStore(base_dir=Path(tmpdir))
        user_id = "test_user"

        store.append_pending(user_id, "- [preference] 用户喜欢测试\n- [identity] 用户是开发者\n")
        result = store.read_pending(user_id)
        assert "[preference] 用户喜欢测试" in result
        assert "[identity] 用户是开发者" in result

        store.append_pending(user_id, "- [key_info] api_key: xxx\n")
        result = store.read_pending(user_id)
        assert "[key_info] api_key: xxx" in result


def test_user_isolation():

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MarkdownMemoryStore(base_dir=Path(tmpdir))

        store.write_long_term("alice", "- [identity] Alice")
        store.write_long_term("bob", "- [identity] Bob")

        assert "Alice" in store.read_long_term("alice")
        assert "Bob" in store.read_long_term("bob")
        assert "Alice" not in store.read_long_term("bob")
        assert "Bob" not in store.read_long_term("alice")


def test_history_append():

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MarkdownMemoryStore(base_dir=Path(tmpdir))
        uid = "test_user"

        store.append_history(uid, "[2026-06-01 10:00] 用户做了A\n")
        store.append_history(uid, "[2026-06-02 15:30] 用户做了B\n")
        result = store.read_history(uid)
        assert "用户做了A" in result
        assert "用户做了B" in result
