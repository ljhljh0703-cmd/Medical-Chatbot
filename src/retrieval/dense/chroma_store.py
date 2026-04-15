"""
ChromaDB 벡터 스토어 래퍼.

저장 메타데이터: c_id, domain, chunk_index, source_spec, token_count
조회 시 유사도 점수(distance) 포함 반환.
"""

from typing import Optional
from observability.logger import logger


class ChromaStore:
    """
    ChromaDB 컬렉션 CRUD 클래스.
    collection_name, persist_directory는 config/settings.py에서 주입.
    """

    def __init__(
        self,
        collection_name: str = "medical_knowledge",
        persist_directory: str = "./chroma_db",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    def _get_collection(self):
        """컬렉션을 lazy 초기화하여 반환."""
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # cosine 유사도 사용
            )
            logger.info(f"[ChromaStore] 컬렉션 '{self.collection_name}' 연결 완료")
        except ImportError:
            raise ImportError("[ChromaStore] chromadb 패키지가 필요합니다.")
        return self._collection

    def save(
        self,
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
        documents: list[str],
    ) -> None:
        """청크 임베딩·메타데이터·원문을 ChromaDB에 저장."""
        col = self._get_collection()
        col.upsert(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            documents=documents,
        )
        logger.info(f"[ChromaStore] {len(ids)}개 청크 upsert 완료")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """
        쿼리 임베딩으로 top_k 유사 청크 검색.

        반환 형식 (chromadb 원본):
        {
            "ids": [[...]],
            "documents": [[...]],
            "metadatas": [[...]],
            "distances": [[...]]   ← cosine distance (낮을수록 유사)
        }
        """
        col = self._get_collection()
        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    def count(self) -> int:
        """컬렉션에 저장된 전체 청크 수 반환."""
        return self._get_collection().count()

    def delete_collection(self) -> None:
        """컬렉션 전체 삭제 (재구축 시 사용)."""
        if self._client:
            self._client.delete_collection(self.collection_name)
            self._collection = None
            logger.warning(f"[ChromaStore] 컬렉션 '{self.collection_name}' 삭제됨")
