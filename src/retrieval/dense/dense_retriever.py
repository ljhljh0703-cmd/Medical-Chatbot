"""
Dense Retriever.

쿼리 텍스트 → 임베딩 → ChromaDB cosine 검색 → RetrievedChunk 리스트 반환.
각 청크에 유사도 점수(similarity) 포함.
"""

from retrieval.dense.embedder import Embedder
from retrieval.dense.chroma_store import ChromaStore
from domain.models.chat_result import RetrievedChunk
from observability.logger import logger


class DenseRetriever:
    """
    Dense retrieval 수행 클래스.
    Embedder와 ChromaStore를 주입받아 사용.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: ChromaStore | None = None,
        top_k: int = 5,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or ChromaStore()
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        query: 정규화된 사용자 질문
        top_k: 검색 결과 수 (None이면 self.top_k 사용)

        반환: RetrievedChunk 리스트 (similarity_score 내림차순)
        """
        k = top_k or self.top_k

        # 1. 쿼리 임베딩
        query_vec = self.embedder.embed(query)

        # 2. ChromaDB 검색
        results = self.store.query(query_embedding=query_vec, top_k=k)

        # 3. 결과 파싱 → RetrievedChunk 조립
        chunks: list[RetrievedChunk] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            logger.warning("[DenseRetriever] 검색 결과 없음")
            return []

        ids        = results["ids"][0]
        documents  = results["documents"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]  # cosine distance: 0=동일, 2=반대

        for doc_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            # cosine distance → similarity 변환 (0~1 스케일)
            similarity = max(0.0, 1.0 - dist / 2.0)
            chunks.append(
                RetrievedChunk(
                    c_id=meta.get("c_id", doc_id),
                    source_spec=meta.get("source_spec") or None,
                    content_snippet=text[:300],  # 최대 300자 미리보기
                    similarity_score=round(similarity, 4),
                )
            )

        # similarity 내림차순 정렬
        chunks.sort(key=lambda c: c.similarity_score, reverse=True)
        logger.info(f"[DenseRetriever] '{query[:30]}...' → {len(chunks)}개 청크 검색")
        return chunks
