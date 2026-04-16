"""
Global configuration module.
Loads environment variables from `.env`.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings
try:
    from pydantic import field_validator as _field_validator

    def before_validator(*fields):
        return _field_validator(*fields, mode="before")
except Exception:  # pragma: no cover - pydantic v1 fallback
    from pydantic import validator as _validator

    def before_validator(*fields):
        return _validator(*fields, pre=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHROMA_DB_PATH = (PROJECT_ROOT / "chroma_db").resolve()


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None

    model_mode: Literal["A", "B", "C"] = "C"
    model_backend: Literal["openai", "qwen"] = "qwen"
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_adapter_path: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    max_new_tokens: int = 512
    temperature: float = 0.7

    embedding_model: str = "jhgan/ko-sroberta-multitask"

    # Keep this path absolute regardless of current working directory.
    chroma_db_path: str = str(DEFAULT_CHROMA_DB_PATH)
    chroma_collection: str = "medical_knowledge"

    top_k: int = 5
    use_hybrid: bool = False
    hybrid_alpha: float = 0.7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @before_validator("chroma_db_path")
    @classmethod
    def _normalize_chroma_db_path(cls, value: Optional[str]) -> str:
        """Normalize relative Chroma path to project-root-based absolute path."""
        if value is None or not str(value).strip():
            return str(DEFAULT_CHROMA_DB_PATH)

        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())


settings = Settings()
