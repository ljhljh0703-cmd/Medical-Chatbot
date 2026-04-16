"""
Chat 오케스트레이터.

전체 파이프라인:
  질문 수신 → Safety 검증(Red Flag) → Retrieval(RAG) → Generation(LLM)
  → Post Filter(톤 보정+면체 조항) → ChatResult 반환
"""

from typing import Optional
from domain.models.chat_result import ChatResult, RetrievedChunk
from services.retrieval_service import RetrievalService
from llm.generation_service import GenerationService
from safety.safety_service import SafetyService
from config.settings import settings
from observability.logger import logger


class ChatService:
    """
    단일 진입점으로, 쿼리를 받아 완성된 ChatResult를 반환.
    Evaluator 및 Streamlit 프론트엔드 모두 이 클래스를 통해 호출한다.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        generation_service: GenerationService | None = None,
        safety_service: SafetyService | None = None,
    ):
        self.retrieval = retrieval_service or RetrievalService()
        self.generation = generation_service or GenerationService()
        self.safety = safety_service or SafetyService()

    def handle(
        self,
        query: str,
        mode: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> ChatResult:
        """
        전체 파이프라인 실행.

        query: 사용자 질문
        mode: "A" | "B" | "C" (버러도 settings.model_mode 사용)
        ground_truth: 테스트 모드 시 라벨링 정답 (None이면 비워둠)
        """
        active_mode = mode or settings.model_mode
        logger.info(f"[ChatService] 요청 수신 | mode={active_mode} | query='{query[:50]}...'")

        # 1. Safety: Red Flag 검사
        red_flag = self.safety.check_query(query)

        # 1-a. Bypass: 즉각 응급 응답 반환 (RAG/LLM 파이프라인 완전 생략)
        if red_flag.bypass:
            logger.warning(f"[ChatService] 🚨 RED FLAG BYPASS | query='{query[:50]}'")
            return ChatResult(
                query=query,
                chatbot_answer=(
                    "🚨 응급 상황이 감지되었습니다.\n\n"
                    "즉시 **119에 연락**하거나 **가장 가까운 응급실로 이동**하십시오."
                ),
                ground_truth=ground_truth,
                retrieved_sources=[],
                red_flag_triggered=True,
                mode=active_mode,
                top_k=settings.top_k,
            )

        # 2. Retrieval: 모드 B/C 일 때만 RAG 검색
        retrieved_chunks: list[RetrievedChunk] = []
        context: Optional[str] = None

        if active_mode in ("B", "C"):
            retrieved_chunks = self.retrieval.retrieve(query)
            context = self.retrieval.format_context(retrieved_chunks)
            logger.info(f"[ChatService] RAG 검색 완료: {len(retrieved_chunks)}개 청크")

        # 3. Generation: LLM 답변 생성
        raw_answer = self.generation.generate(
            query=query,
            context=context,
            mode=active_mode,
        )

        # 4. Post Filter: 톤 보정 + 면체 조항 + Red Flag 배너
        final_answer = self.safety.process_response(
            response=raw_answer,
            red_flag_triggered=red_flag.triggered,
        )

        # 5. ChatResult 조립
        result = ChatResult(
            query=query,
            chatbot_answer=final_answer,
            ground_truth=ground_truth,
            retrieved_sources=retrieved_chunks,
            red_flag_triggered=red_flag.triggered,
            mode=active_mode,
            top_k=settings.top_k,
        )

        logger.info(f"[ChatService] 응답 완료 | red_flag={red_flag.triggered} | sources={len(retrieved_chunks)}")
        return result
