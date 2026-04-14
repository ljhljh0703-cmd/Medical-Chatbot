"""
Hybrid Retrieval — Dense + BM25 점수 융합 (RRF / 가중 합산).

Dense(코사인 유사도) + BM25(키워드 점수)를 결합하여
단독 검색보다 더 균형 잡힌 결과 제공.

융합 방식:
  - weighted_sum : alpha * dense_score + (1-alpha) * bm25_normalized
  - rrf          : Reciprocal Rank Fusion (순위 기반, 파라미터 적음)
"""

from __future__ import annotations

from domain.models.chat_result import RetrievedChunk
from retrieval.dense.dense_retriever import DenseRetriever
from retrieval.sparse.bm25_retriever import BM25Retriever
from observability.logger import logger


class HybridFusion:
    """
    Dense + Sparse 하이브리드 검색기.

    사용 예:
        fusion = HybridFusion(alpha=0.7)
        chunks = fusion.retrieve("당뇨병 합병증", top_k=5)
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        alpha: float = 0.7,                     # Dense 가중치 (1-alpha = BM25)
        fusion_method: str = "weighted_sum",    # "weighted_sum" | "rrf"
        rrf_k: int = 60,                        # RRF 상수
    ) -> None:
        self.dense = dense_retriever or DenseRetriever()
        self.bm25 = bm25_retriever or BM25Retriever()
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

    # ── 메인 검색 ──────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Dense + BM25 결합 검색."""
        dense_results: list[RetrievedChunk] = self.dense.retrieve(
            query, top_k=top_k * 2
        )
        bm25_results: list[dict] = self.bm25.retrieve(query, top_k=top_k * 2)

        if self.fusion_method == "rrf":
            fused = self._rrf_fusion(dense_results, bm25_results, top_k)
        else:
            fused = self._weighted_sum_fusion(dense_results, bm25_results, top_k)

        logger.info(
            f"[HybridFusion] 융합 완료: dense={len(dense_results)}, "
            f"bm25={len(bm25_results)} → 최종={len(fused)}"
        )
        return fused

    # ── 가중 합산 ──────────────────────────────────────────────

    def _weighted_sum_fusion(
        self,
        dense: list[RetrievedChunk],
        bm25: list[dict],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """alpha * dense_score + (1-alpha) * bm25_normalized."""
        bm25_scores = {r["id"]: r["bm25_score"] for r in bm25}
        max_bm25 = max(bm25_scores.values(), default=1.0) or 1.0

        score_map: dict[str, dict] = {}
        for chunk in dense:
            score_map[chunk.c_id] = {
                "chunk": chunk,
                "dense": chunk.similarity_score,
                "bm25": bm25_scores.get(chunk.c_id, 0.0) / max_bm25,
            }

        # BM25 전용 결과 추가 (Dense 미포함 문서)
        for r in bm25:
            cid = r["id"]
            if cid not in score_map:
                score_map[cid] = {
                    "chunk": RetrievedChunk(
                        c_id=cid,
                        source_spec=r["metadata"].get("source_spec"),
                        content_snippet=r["text"][:300],
                        similarity_score=0.0,
                    ),
                    "dense": 0.0,
                    "bm25": r["bm25_score"] / max_bm25,
                }

        scored = []
        for item in score_map.values():
            fused_score = (
                self.alpha * item["dense"] + (1 - self.alpha) * item["bm25"]
            )
            chunk = item["chunk"].model_copy(
                update={"similarity_score": fused_score}
            )
            scored.append(chunk)

        scored.sort(key=lambda c: c.similarity_score, reverse=True)
        return scored[:top_k]

    # ── RRF 융합 ───────────────────────────────────────────────

    def _rrf_fusion(
        self,
        dense: list[RetrievedChunk],
        bm25: list[dict],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion: score += 1 / (k + rank)."""
        rrf_scores: dict[str, float] = {}

        for rank, chunk in enumerate(dense, 1):
            rrf_scores[chunk.c_id] = (
                rrf_scores.get(chunk.c_id, 0.0) + 1.0 / (self.rrf_k + rank)
            )
        for rank, r in enumerate(bm25, 1):
            cid = r["id"]
            rrf_scores[cid] = (
                rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            )

        dense_map = {c.c_id: c for c in dense}
        bm25_map = {r["id"]: r for r in bm25}

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        result = []
        for cid, score in ranked:
            if cid in dense_map:
                chunk = dense_map[cid].model_copy(
                    update={"similarity_score": score}
                )
            else:
                r = bm25_map[cid]
                chunk = RetrievedChunk(
                    c_id=cid,
                    source_spec=r["metadata"].get("source_spec"),
                    content_snippet=r["text"][:300],
                    similarity_score=score,
                )
            result.append(chunk)
        return result
