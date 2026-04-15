"""
BM25 ?ъ냼(Sparse) 寃?됯린.

rank-bm25 ?⑦궎吏 湲곕컲. ChromaDB?먯꽌 ?꾩껜 corpus瑜?濡쒕뱶?섏뿬
BM25 ?몃뜳?ㅻ? 硫붾え由ъ뿉 援ъ텞?섍퀬 荑쇰━ ?좏겙 湲곕컲?쇰줈 寃??

requirements: rank-bm25  (pip install rank-bm25)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from observability.logger import logger

DEFAULT_CHROMA_DB_PATH = (Path(__file__).resolve().parents[3] / "chroma_db").resolve()

"""
?곕????ㅽ뻾 肄붾뱶:
pip install kiwipiepy rank-bm25 chromadb # ?쇱씠釉뚮윭由??ㅼ튂
python -c "import chromadb; c=chromadb.PersistentClient(path='../chroma_db'); print([x.name for x in c.list_collections()])" # chroma_db 而щ젆??濡쒕뱶
python -m ingestion.indexing.build_knowledge_base --data_dir ../data/raw --collection medical_knowledge # chroma_db 而щ젆??濡쒕뱶媛 ?덈맆 寃쎌슦 ?앹꽦
python -c "from retrieval.sparse.bm25_retriever import BM25Retriever; r=BM25Retriever(collection_name='medical_knowledge', db_path='../chroma_db'); print(r.retrieve('怨좏삁??移섎즺', top_k=5))" # bm25 index ?앹꽦
"""

class BM25Retriever:
    """
    BM25 寃?됯린.

    ?ъ슜 ??
        retriever = BM25Retriever(collection_name="medical_knowledge")
        results = retriever.retrieve("怨좏삁??移섎즺", top_k=5)
    """

    _DOMAIN_TOKEN_PATTERN = re.compile(
        r"\b(?:[A-Z]{2,}[A-Z0-9]*|[A-Za-z]+[0-9]+[A-Za-z0-9]*|[A-Za-z0-9]+(?:[+\-_/][A-Za-z0-9]+)+|[0-9]+(?:\.[0-9]+)?(?:mg|g|mcg|ml|l|mmhg|mmol/?l|mg/?dl|iu|u|%)?)\b",
        flags=re.IGNORECASE,
    )
    _KOREAN_WORD_PATTERN = re.compile(r"[媛-??{2,}")
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
        """kiwipiepy.Kiwi lazy 濡쒕뱶. 誘몄꽕移???None 諛섑솚."""
        if self._kiwi_checked:
            return self._kiwi

        self._kiwi_checked = True
        try:
            from kiwipiepy import Kiwi  # type: ignore

            self._kiwi = Kiwi()
            logger.info("[BM25Retriever] Kiwi ?뺥깭??遺꾩꽍湲?濡쒕뱶 ?꾨즺")
        except Exception as exc:
            self._kiwi = None
            logger.warning(
                f"[BM25Retriever] Kiwi 濡쒕뱶 ?ㅽ뙣, regex ?좏겙?붾줈 fallback: {exc}"
            )
        return self._kiwi

    def _tokenize(self, text: str) -> list[str]:
        """
        ?섎즺 ?꾨찓???좏겙? regex濡??좎텛異?蹂댄샇?섍퀬,
        ?섎㉧吏 ?쒓?? Kiwi ?뺥깭??遺꾩꽍?쇰줈 ?좏겙??
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
                    f"[BM25Retriever] Kiwi 遺꾩꽍 ?ㅽ뙣, regex ?쒓? ?좏겙 fallback: {exc}"
                )
                tokens.extend(
                    t.lower() for t in self._KOREAN_WORD_PATTERN.findall(remaining)
                )
        else:
            tokens.extend(t.lower() for t in self._KOREAN_WORD_PATTERN.findall(remaining))

        if not tokens:
            tokens = [t.lower() for t in normalized.split() if t.strip()]
        return tokens

    # ?? ?몃뜳??援ъ텞 ????????????????????????????????????????????

    def _build_index(self) -> None:
        """ChromaDB?먯꽌 ?꾩껜 corpus瑜?遺덈윭? BM25 ?몃뜳??援ъ텞."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 ?⑦궎吏媛 ?꾩슂?⑸땲?? pip install rank-bm25"
            ) from exc

        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.db_path)
            collection = client.get_collection(self.collection_name)
            result = collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            logger.error(f"[BM25Retriever] ChromaDB 濡쒕뱶 ?ㅽ뙣: {exc}")
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
            f"[BM25Retriever] ?몃뜳??援ъ텞 ?꾨즺: {len(self._corpus_docs)}媛?臾몄꽌"
        )

    # ?? 寃?????????????????????????????????????????????????????

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        BM25 寃??

        諛섑솚媛?
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
        """?몃뜳??媛뺤젣 ?ш뎄異?(?좉퇋 臾몄꽌 異붽? ???몄텧)."""
        self._bm25 = None
        self._corpus_docs = []
        self._build_index()
