# -*- coding: utf-8 -*-

"""
BM25 희소(Sparse) 검색기.

rank-bm25 패키지 기반. ChromaDB에서 전체 corpus를 로드하여
BM25 인덱스를 메모리에 구축하고 쿼리 토큰 기반으로 검색.

requirements: rank-bm25  (pip install rank-bm25)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from observability.logger import logger

DEFAULT_CHROMA_DB_PATH = (Path(__file__).resolve().parents[3] / "chroma_db").resolve()

"""
로컬 실행 코드:
pip install kiwipiepy rank-bm25 chromadb # 라이브러리 설치
python -c "import chromadb; c=chromadb.PersistentClient(path='../chroma_db'); print([x.name for x in c.list_collections()])" # chroma_db 컬렉션 로드
python -m ingestion.indexing.build_knowledge_base --data_dir ../data/raw --collection medical_knowledge # chroma_db 컬렉션이 없을 경우 생성
python -c "from retrieval.sparse.bm25_retriever import BM25Retriever; r=BM25Retriever(collection_name='medical_knowledge', db_path='../chroma_db'); print(r.retrieve('고혈압 치료', top_k=5))" # bm25 index 생성
"""

class BM25Retriever:
    """
    BM25 검색기.

    사용 예:
        retriever = BM25Retriever(collection_name="medical_knowledge")
        results = retriever.retrieve("고혈압 치료", top_k=5)
    """

    _DOMAIN_TOKEN_PATTERN = re.compile(
        r"\b(?:[A-Z]{2,}[A-Z0-9]*|[A-Za-z]+[0-9]+[A-Za-z0-9]*|[A-Za-z0-9]+(?:[+\-_/][A-Za-z0-9]+)+|[0-9]+(?:\.[0-9]+)?(?:mg|g|mcg|ml|l|mmhg|mmol/?l|mg/?dl|iu|u|%)?)\b",
        flags=re.IGNORECASE,
    )
    _KOREAN_WORD_PATTERN = re.compile(r"[가-힣]{2,}")
    _KIWI_ALLOWED_POS = {"NNG", "NNP", "SL", "SN", "XR"}

    def __init__(
        self,
        collection_name: str = "medical_knowledge",
        db_path: str = str(DEFAULT_CHROMA_DB_PATH),
    ) -> None:
        self.collection_name = collection_name
        self.db_path = db_path
        self._bm25 = None
        self._corpus_docs: list[dict] = []   # {id, text, metadata}
        self._kiwi: Any | None = None
        self._kiwi_checked = False

    def _get_kiwi(self) -> Any | None:
        """kiwipiepy.Kiwi lazy 로드. 미설치 시 None 반환."""
        if self._kiwi_checked:
            return self._kiwi

        self._kiwi_checked = True
        try:
            from kiwipiepy import Kiwi  # type: ignore

            self._kiwi = Kiwi()
            logger.info("[BM25Retriever] Kiwi 형태소 분석기 로드 완료")
        except Exception as exc:
            self._kiwi = None
            logger.warning(
                f"[BM25Retriever] Kiwi 로드 실패, regex 토큰화로 fallback: {exc}"
            )
        return self._kiwi

    def _tokenize(self, text: str) -> list[str]:
        """
        도메인 패턴 토큰은 regex로 추출해 보존하고,
        나머지 텍스트는 Kiwi 형태소 분석으로 토큰화.
        """
        if not text or not text.strip():
            return []

        normalized = re.sub(r"\s+", " ", text).strip()
        domain_tokens = [
            m.group(0).lower() for m in self._DOMAIN_TOKEN_PATTERN.finditer(normalized)
        ]
        remaining = self._DOMAIN_TOKEN_PATTERN.sub(" ", normalized)

        tokens: list[str] = list(domain_tokens)
        kiwi = self._get_kiwi()

        if kiwi is not None:
            try:
                for token in kiwi.tokenize(remaining):
                    surface = (token.form or "").strip().lower()
                    if len(surface) < 2:
                        continue
                    if token.tag not in self._KIWI_ALLOWED_POS:
                        continue
                    if not self._KOREAN_WORD_PATTERN.fullmatch(surface):
                        continue
                    tokens.append(surface)
            except Exception as exc:
                logger.warning(
                    f"[BM25Retriever] Kiwi 분석 실패, regex 한국어 토큰 fallback: {exc}"
                )
                tokens.extend(
                    t.lower() for t in self._KOREAN_WORD_PATTERN.findall(remaining)
                )
        else:
            tokens.extend(t.lower() for t in self._KOREAN_WORD_PATTERN.findall(remaining))

        if not tokens:
            tokens = [t.lower() for t in normalized.split() if t.strip()]
        return tokens

    # 인덱스 구축
    
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

        tokenized = [self._tokenize(doc["text"]) for doc in self._corpus_docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            f"[BM25Retriever] 인덱스 구축 완료: {len(self._corpus_docs)}개 문서"
        )

    # 검색

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

        tokens = self._tokenize(query)
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
        """인덱스 강제 재구축(신규 문서 추가 후 호출)."""
        self._bm25 = None
        self._corpus_docs = []
        self._build_index()
