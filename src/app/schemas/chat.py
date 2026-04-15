"""
FastAPI 요청/응답 스키마.

- ChatRequest  : 클라이언트 → 서버
- ChatResponse : 서버 → 클라이언트 (ChatResult 직렬화)
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="사용자 질문")
    mode: Optional[str] = Field(
        default=None,
        description="생성 모드 A/B/C. 미입력 시 settings.model_mode 사용",
    )
    ground_truth: Optional[str] = Field(
        default=None,
        description="라벨링 정답 (평가/테스트 시 선택 입력)",
    )


class RetrievedChunkSchema(BaseModel):
    c_id: str
    source_spec: Optional[str] = None
    content_snippet: str
    similarity_score: float


class ChatResponse(BaseModel):
    query: str
    chatbot_answer: str
    ground_truth: Optional[str] = None
    retrieved_sources: list[RetrievedChunkSchema] = []
    red_flag_triggered: bool = False
    mode: str


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"

