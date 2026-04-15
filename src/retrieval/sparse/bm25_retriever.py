"""
BM25 희소(Sparse) 검색기.

rank-bm25 패키지 기반. ChromaDB에서 전체 corpus를 로드하여
BM25 인덱스를 메모리에 구축하고 쿼리 토큰 기반으로 검색.

requirements: rank-bm25  (pip install rank-bm25)
"""

from __future__ import annotations

from observability.logger import logger


class BM25Retriever:
    """
    BM25 검색기.

    사용 예:
        retriever = BM25Retriever(collection_name="medical_knowledge")
        results = retriever.retrieve("고혈압 치료", top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "medical_knowledge",
        db_path: str = "./chroma_db",
    ) -> None:
        self.collection_name = collection_name
        self.db_path = db_path
        self._bm25 = None
        self._corpus_docs: list[dict] = []   # {id, text, metadata}

    # ── 인덱스 구축 ────────────────────────────────────────────

    def _build_index(self) -> None:
        """ChromaDB에서 전체 corpus를 불러와 BM25 인덱스 구축."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 패키지가 필요합니다: pip install rank-bm25"
            ) from exc

        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.db_path)
            collection = client.get_collection(self.collection_name)
            result = collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            logger.error(f"[BM25Retriever] ChromaDB 로드 실패: {exc}")
            self._bm25 = None
            return

        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", []) or [{}] * len(docs)

        self._corpus_docs = [
            {"id": i, "text": d, "metadata": m}
            for i, d, m in zip(ids, docs, metas)
        ]

        tokenized = [doc["text"].split() for doc in self._corpus_docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            f"[BM25Retriever] 인덱스 구축 완료: {len(self._corpus_docs)}개 문서"
        )

    # ── 검색 ───────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        BM25 검색.

        반환값:
            list of {id, text, metadata, bm25_score}
        """
        if self._bm25 is None:
            self._build_index()
        if self._bm25 is None or not self._corpus_docs:
            return []

        tokens = query.split()
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            if score < min_score:
                continue
            doc = self._corpus_docs[idx]
            results.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "bm25_score": float(score),
                }
            )
        return results

    def rebuild_index(self) -> None:
        """인덱스 강제 재구축 (신규 문서 추가 후 호출)."""
        self._bm25 = None
        self._corpus_docs = []
        self._build_index()
