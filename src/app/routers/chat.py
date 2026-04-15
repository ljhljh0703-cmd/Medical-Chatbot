"""
Chat API 라우터.

Endpoints:
  POST /chat/message  — 챗봇 응답 생성
  GET  /chat/health   — 서비스 상태 확인
"""

from functools import lru_cache
from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse, HealthResponse

router = APIRouter()


@lru_cache(maxsize=1)
def _get_chat_service():
    """ChatService 싱글턴 (첫 요청 시 lazy 초기화)."""
    try:
        from services.chat_service import ChatService  # noqa: WPS433
        return ChatService()
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"ChatService 초기화 실패: {exc}") from exc


@router.post("/message", response_model=ChatResponse, summary="챗봇 질의")
def send_message(request: ChatRequest) -> ChatResponse:
    """
    사용자 질문을 받아 챗봇 응답을 반환합니다.

    - **query**: 사용자 질문 (필수)
    - **mode**: 생성 모드 A/B/C (선택, 기본값: settings.model_mode)
    - **ground_truth**: 라벨링 정답 (선택, 평가용)
    """
    try:
        svc = _get_chat_service()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = svc.handle(
            query=request.query,
            mode=request.mode,
            ground_truth=request.ground_truth,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"생성 오류: {exc}") from exc

    return ChatResponse(
        query=result.query,
        chatbot_answer=result.chatbot_answer,
        ground_truth=result.ground_truth,
        retrieved_sources=[
            {"c_id": s.c_id, "source_spec": s.source_spec,
             "content_snippet": s.content_snippet, "similarity_score": s.similarity_score}
            for s in result.retrieved_sources
        ],
        red_flag_triggered=result.red_flag_triggered,
        mode=result.mode,
    )


@router.get("/health", response_model=HealthResponse, summary="서비스 상태 확인")
def health_check() -> HealthResponse:
    """서비스 정상 동작 여부를 반환합니다."""
    return HealthResponse(status="ok")

