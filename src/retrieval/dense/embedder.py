"""
임베딩 모듈.

기본: OpenAI text-embedding-3-small (config/settings.py 에서 모델 교체 가능)
Fallback: sentence-transformers 로컬 모델 (API 키 없을 때 자동 전환)
"""

from typing import Optional
from observability.logger import logger


class Embedder:
    """
    텍스트 → 벡터 변환 클래스.

    embedding_model 값:
      - "openai:<model>"  예) "openai:text-embedding-3-small"
      - "local:<model>"   예) "local:snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    """

    OPENAI_DEFAULT = "text-embedding-3-small"
    LOCAL_DEFAULT  = "jhgan/ko-sroberta-multitask"  # 한국어 특화 SBERT

    def __init__(self, embedding_model: str = "jhgan/ko-sroberta-multitask", api_key: Optional[str] = None):
        self.embedding_model = embedding_model
        self.api_key = api_key
        self._local_model = None

    # ── 내부 헬퍼 ──────────────────────────────────────────────

    def _embed_openai(self, texts: list[str], model: str) -> list[list[float]]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.warning(f"[Embedder] OpenAI 임베딩 실패, 로컬로 전환: {e}")
            return self._embed_local(texts, self.LOCAL_DEFAULT)

    def _embed_local(self, texts: list[str], model: str) -> list[list[float]]:
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(model)
                logger.info(f"[Embedder] 로컬 모델 로드 완료: {model}")
            except ImportError:
                raise ImportError("[Embedder] sentence-transformers 패키지가 필요합니다.")
        vecs = self._local_model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    # ── 퍼블릭 API ─────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """단일 텍스트 임베딩."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """텍스트 리스트 일괄 임베딩."""
        if not texts:
            return []

        prefix, _, model_name = self.embedding_model.partition(":")
        model_name = model_name or self.OPENAI_DEFAULT

        if prefix == "openai":
            return self._embed_openai(texts, model_name)
        else:  # "local" 또는 미지정
            return self._embed_local(texts, model_name or self.LOCAL_DEFAULT)
