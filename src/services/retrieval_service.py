"""
Retrieval 오케스트레이터.

QueryNormalizer → DenseRetriever → RetrievedChunk 리스트 반환.
RAG 컨텍스트 문자열 조립 편의 메서드 제공.
"""

from retrieval.query.normalizer import QueryNormalizer
from retrieval.dense.dense_retriever import DenseRetriever
from retrieval.dense.embedder import Embedder
from retrieval.dense.chroma_store import ChromaStore
from domain.models.chat_result import RetrievedChunk
from config.settings import settings
from observability.logger import logger


class RetrievalService:
    """
    검색 전체 흐름을 관리하는 서비스 계층.
    외부에서 컨텍스트 문자열만 필요하면 format_context() 를 호출하면 된다.
    """

    def __init__(
        self,
        normalizer: QueryNormalizer | None = None,
        retriever: DenseRetriever | None = None,
    ):
        self.normalizer = normalizer or QueryNormalizer()
        self.retriever = retriever or DenseRetriever(
            embedder=Embedder(
                embedding_model=settings.embedding_model,
                api_key=settings.openai_api_key,
            ),
            store=ChromaStore(
                collection_name=settings.chroma_collection,
                persist_directory=settings.chroma_db_path,
            ),
            top_k=settings.top_k,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """쿼리 정규화 → Dense 검색 → RetrievedChunk 리스트 반환."""
        normalized = self.normalizer.normalize(query)
        logger.info(f"[RetrievalService] 정규화: '{query[:40]}' → '{normalized[:40]}'")
        return self.retriever.retrieve(normalized, top_k=top_k)

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        RetrievedChunk 리스트를 RAG 컨텍스트 문자열로 조립.
        GenerationService._build_prompt() 에 전달할 용도.
        """
        if not chunks:
            return ""
        parts = []
        for i, c in enumerate(chunks, 1):
            source = c.source_spec or "미상"
            parts.append(
                f"[참고 {i}] (출싸: {source}, 유사도: {c.similarity_score:.2f})\n"
                f"{c.content_snippet}"
            )
        return "\n\n".join(parts)
