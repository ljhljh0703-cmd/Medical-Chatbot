"""
임베딩 모듈.

기본: OpenAI text-embedding-3-small (config/settings.py 에서 모델 교체 가능)
Fallback: sentence-transformers 로컬 모델 (API 키 없을 때 자동 전환)
"""

import os
import logging
from typing import List, Optional

# Attempt to import SentenceTransformer, fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence-transformers library not found. Local embedding will be disabled.")

# Attempt to import OpenAI client, fallback if not available
try:
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False
    logging.warning("OpenAI client not found. OpenAI embedding will be disabled.")

logger = logging.getLogger(__name__)

class Embedder:
    """
    텍스트 → 벡터 변환 클래스.

    embedding_model 값:
      - "openai:<model>"  예) "openai:text-embedding-3-small"
      - "local:<model>"   예) "local:snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    """
    def __init__(self, embedding_model: str = "jhgan/ko-sroberta-multitask", api_key: Optional[str] = None):
        self.api_key = api_key
        self.embedding_model = embedding_model # This should be the default local model
        self.client = None
        self.st_model = None

        if self.api_key and OPENAI_CLIENT_AVAILABLE:
            try:
                self.client = OpenAI(api_key=self.api_key)
                # If the specified embedding_model is not an OpenAI model, use a sensible default for OpenAI
                if not self.embedding_model.startswith("text-embedding"):
                    self.openai_model_name = "text-embedding-3-small" # Default OpenAI model
                else:
                    self.openai_model_name = self.embedding_model # Use the specified model if it's an OpenAI model name
                logger.info(f"[Embedder] Using OpenAI embedding model: {self.openai_model_name}")
            except Exception as e:
                logger.warning(f"[Embedder] OpenAI 임베딩 실패, 로컬로 전환: {e}")
                self.client = None # Fallback
        
        if not self.client and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # IMPORTANT: Use the self.embedding_model here for local model initialization
                self.st_model = SentenceTransformer(self.embedding_model)
                logger.info(f"[Embedder] Using local sentence-transformers model: {self.embedding_model}")
            except Exception as e:
                logger.error(f"[Embedder] Failed to load local sentence-transformers model '{self.embedding_model}': {e}")
                raise

        if not self.client and not self.st_model:
            raise RuntimeError("No embedding method could be initialized. Please check API key or local model availability.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """단일 & 배치 텍스트 임베딩."""
        if self.client:
            response = self.client.embeddings.create(input=texts, model=self.openai_model_name)
            return [d.embedding for d in response.data]
        elif self.st_model:
            return self.st_model.encode(texts).tolist()
        else:
            raise RuntimeError("Embedding method not initialized.")