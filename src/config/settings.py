"""
전역 설정 모듈.

환경 변수 또는 .env 파일에서 로드.
"""

from pydantic_settings import BaseSettings
from typing import Literal, Optional


class Settings(BaseSettings):
    # ── API 키 ───────────────────────────────────────────
    openai_api_key: Optional[str] = None

    # ── 모델 설정 ─────────────────────────────────────────
    model_mode: Literal["A", "B", "C"] = "C"          # A=LLM only, B=LLM+RAG, C=LLM+RAG+LoRA
    model_backend: Literal["openai", "qwen"] = "qwen"  # 추론 백엔드
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"       # 베이스 모델 (HF ID 또는 로컬 경로)
    lora_adapter_path: Optional[str] = None              # LoRA 어댑터 경로 (모드 C 전용)
    openai_model: str = "gpt-4o-mini"                   # OpenAI 백엔드 사용 시 모델명
    max_new_tokens: int = 512
    temperature: float = 0.7

    # ── 임베딩 ──────────────────────────────────────────
    embedding_model: str = "openai:text-embedding-3-small"

    # ── ChromaDB ─────────────────────────────────────────
    chroma_db_path: str = "./chroma_db"
    chroma_collection: str = "medical_inner"

    # ── 검색 ───────────────────────────────────────────
    top_k: int = 5
    use_hybrid: bool = False          # True = BM25+Dense 혼합 검색
    hybrid_alpha: float = 0.7        # Dense 가중치 (1-alpha = BM25 가중치)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
