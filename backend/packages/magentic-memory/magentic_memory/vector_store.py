import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class VectorMemoryStore:
    """Chroma 向量记忆存储。四类记忆各用一个 collection。

    Collections:
        memory_events      — 时间线事件
        memory_procedures  — 操作规范
        memory_preferences — 用户偏好
        memory_profiles    — 用户画像
    """

    _COLLECTION_NAMES = {
        "event": "memory_events",
        "procedure": "memory_procedures",
        "preference": "memory_preferences",
        "profile": "memory_profiles",
    }

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
        )
        self._collections: dict[str, Chroma] = {}
        for mtype, col_name in self._COLLECTION_NAMES.items():
            self._collections[mtype] = Chroma(
                client=self._client,
                collection_name=col_name,
                embedding_function=self._embeddings,
            )

    @property
    def collections(self) -> dict[str, Chroma]:
        return self._collections

    def add_memory(
        self,
        memory_type: str,
        content: str,
        metadata: dict | None = None,
        doc_id: str | None = None,
    ):
        """写入一条记忆到对应类型的 collection。"""
        import hashlib

        collection = self._collections[memory_type]
        if doc_id is None:
            doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        meta = {"memory_type": memory_type}
        if metadata:
            meta.update(metadata)

        collection.add_texts(
            texts=[content],
            metadatas=[meta],
            ids=[doc_id],
        )

    def similarity_search(
        self,
        query: str,
        memory_type: str,
        k: int = 5,
        filter: dict | None = None,
    ) -> list[tuple[str, float]]:
        """语义搜索，返回 (文档内容, 相似度分数) 列表。"""
        collection = self._collections[memory_type]
        search_filter = {"memory_type": memory_type}
        if filter:
            search_filter.update(filter)

        results = collection.similarity_search_with_score(
            query, k=k, filter=search_filter
        )
        return [(doc.page_content, score) for doc, score in results]
