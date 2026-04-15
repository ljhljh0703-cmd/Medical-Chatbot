"""
LLM 답변 생성 서비스.

3가지 모드 지원:
  A: LLM only       (베이스 모델만 사용, RAG 없음)
  B: LLM + RAG      (베이스 모델 + 검색된 컨텍스트 주입)
  C: LLM + RAG + LoRA (베이스 + LoRA 어댑터 + 컨텍스트 주입)

백엔드: "openai" 또는 "qwen" (settings.model_backend)
"""

from typing import Optional
from config.prompts import SYSTEM_PROMPT, RAG_CONTEXT_TEMPLATE
from config.settings import settings
from observability.logger import logger


class GenerationService:
    """
    LLM 답변 생성 오케스트레이터.

    mode 오버라이드: generate() 호출 시 mode 매개변수로 변경 가능.
    settings.model_mode 값을 기본값으로 사용.
    """

    def __init__(self):
        self._qwen_base: object | None = None   # 모드 A/B용 베이스 클라이언트
        self._qwen_lora: object | None = None   # 모드 C용 LoRA 클라이언트
        self._openai: object | None = None       # OpenAI 백엔드 클라이언트

    # ── 클라이언트 lazy init ────────────────────────────────────

    def _get_openai(self):
        if self._openai is None:
            from llm.model_clients.openai_client import OpenAIClient
            self._openai = OpenAIClient(
                api_key=settings.openai_api_key,
                model_name=settings.openai_model,
            )
        return self._openai

    def _get_qwen_base(self):
        if self._qwen_base is None:
            from llm.model_clients.qwen_client import QwenClient
            self._qwen_base = QwenClient(model_path=settings.model_path)
            logger.info(f"[GenerationService] Qwen base 로드: {settings.model_path}")
        return self._qwen_base

    def _get_qwen_lora(self):
        """모드 C: LoRA 어댑터 병합된 모델 또는 PEFT 어댑터 로드."""
        if self._qwen_lora is None:
            from llm.model_clients.qwen_client import QwenClient
            adapter = settings.lora_adapter_path
            if adapter:
                # 병합된 모델 또는 어댑터 경로를 model_path로 지정
                self._qwen_lora = QwenClient(model_path=adapter)
                logger.info(f"[GenerationService] Qwen LoRA 로드: {adapter}")
            else:
                logger.warning("[GenerationService] lora_adapter_path 미설정 → base 모델로 fallback")
                self._qwen_lora = self._get_qwen_base()
        return self._qwen_lora

    # ── 프롬프트 조립 ────────────────────────────────────────

    def _build_prompt(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """RAG 컨텍스트 유무에 따라 프롬프트를 조립하여 반환."""
        if context:
            return RAG_CONTEXT_TEMPLATE.format(context=context) + f"\n질문: {query}"
        return query

    # ── 메인 API ───────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> str:
        """
        답변 생성.

        query: 사용자 질문
        context: RAG 검색 컨텍스트 (모드 B/C 전용, 모드 A에서는 무시)
        mode: "A" | "B" | "C" (None이면 settings.model_mode 사용)
        """
        active_mode = mode or settings.model_mode

        # 모드 A: context 무시
        effective_context = context if active_mode in ("B", "C") else None
        prompt = self._build_prompt(query, effective_context)

        logger.info(f"[GenerationService] mode={active_mode}, backend={settings.model_backend}")

        # 백엔드 선택
        if settings.model_backend == "openai":
            client = self._get_openai()
            return client.request(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
            )

        # Qwen 백엔드
        if active_mode == "C":
            client = self._get_qwen_lora()
        else:
            client = self._get_qwen_base()

        return client.request(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
        )
