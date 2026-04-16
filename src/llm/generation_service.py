"""
LLM 답변 생성 서비스.

3가지 모드 지원:
  A: LLM only       (베이스 모델만 사용, RAG 없음)
  B: LLM + RAG      (베이스 모델 + 검색된 컨텍스트 주입)
  C: LLM + RAG + LoRA (베이스 + LoRA 어댑터 + 컨텍스트 주입)

백엔드: "openai" 또는 "qwen" (settings.model_backend)

[주요 수정 내역 (Changelog -  정현)]
1. 클라이언트 호출 파라미터 변경: 새롭게 업그레이드된 QwenClient 구조에 맞추어, 단일 `model_path`가 아닌 `base_model_path`와 `adapter_path`를 명시적으로 분리하여 주입하도록 수정했습니다.
2. 모드 C (LoRA) 로직 개선: settings에서 베이스 모델 경로와 LoRA 경로를 동시에 가져와서 QwenClient에 넘겨주도록 구조를 개편했습니다.


"""
from typing import Optional
from config.prompts import SYSTEM_PROMPT, RAG_CONTEXT_TEMPLATE
from config.settings import settings
from observability.logger import logger

class GenerationService:
    """
    [클래스 개요]
    LLM 답변 생성 오케스트레이터 (통제실).
    사용자의 질문이 들어오면 설정(Mode)에 따라 OpenAI로 보낼지, Qwen Base로 보낼지, Qwen LoRA로 보낼지 결정하고 실행합니다.
    """

    def __init__(self):
        # 지연 초기화(Lazy Init)를 위한 빈 공간 준비
        # 처음부터 무거운 모델을 다 띄우지 않고, 실제로 해당 모드가 호출될 때만 객체를 생성
        self._qwen_base: object | None = None   # 모드 A/B용 베이스 클라이언트
        self._qwen_lora: object | None = None   # 모드 C용 LoRA 클라이언트
        self._openai: object | None = None      # OpenAI 백엔드 클라이언트

    # ── 클라이언트 lazy init (필요할 때만 로드) ────────────────────────────────────

    def _get_openai(self):
        """OpenAI 클라이언트 로드"""
        if self._openai is None:
            from llm.model_clients.openai_client import OpenAIClient
            self._openai = OpenAIClient(
                api_key=settings.openai_api_key,
                model_name=settings.openai_model,
            )
        return self._openai

    def _get_qwen_base(self):
        """모드 A, B: 순수 베이스 모델 로드"""
        if self._qwen_base is None:
            from llm.model_clients.qwen_client import QwenClient
            # 수정됨: base_model_path 파라미터 사용 (어댑터는 None)
            self._qwen_base = QwenClient(
                base_model_path=settings.model_path,
                adapter_path=None
            )
            logger.info(f"[GenerationService] Qwen base 로드 완료: {settings.model_path}")
        return self._qwen_base

    def _get_qwen_lora(self):
        """모드 C: 동적 LoRA 어댑터가 부착된 모델 로드"""
        if self._qwen_lora is None:
            from llm.model_clients.qwen_client import QwenClient
            
            # 설정 파일(.env 또는 settings.py)에서 두 가지 경로를 모두 가져옵니다.
            base_path = settings.model_path
            adapter = settings.lora_adapter_path
            
            if adapter:
                # 💡 수정됨: 뼈대 경로와 메모지(LoRA) 경로를 동시에 주입합니다!
                # 이렇게 하면 QwenClient 내부에서 베이스를 띄우고 그 위에 LoRA를 찰싹 붙여줍니다.
                self._qwen_lora = QwenClient(
                    base_model_path=base_path,
                    adapter_path=adapter
                )
                logger.info(f"[GenerationService] Qwen LoRA 동적 로드 완료 (Base: {base_path} + LoRA: {adapter})")
            else:
                # 사용자가 모드 C를 켰는데 .env에 LoRA 경로를 안 적어뒀을 경우의 안전장치(Fallback)
                logger.warning("[GenerationService] 🚨 lora_adapter_path 미설정! → 안전을 위해 Base 모델로 우회(Fallback)합니다.")
                self._qwen_lora = self._get_qwen_base()
                
        return self._qwen_lora

    # ── 프롬프트 조립 ────────────────────────────────────────

    def _build_prompt(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """
        [기능] RAG 컨텍스트 유무에 따라 최종 프롬프트를 조립합니다.
        문서 검색 결과(context)가 있으면 템플릿에 넣어서 질문과 합치고, 없으면 순수 질문만 반환합니다.
        """
        if context:
            #검색 된 문서를 템플릿에 끼워 넣고 사용자 질문과 합침
            return RAG_CONTEXT_TEMPLATE.format(context=context) + f"\n질문: {query}"
        return query

    # ── 메인 API (외부에서 호출하는 유일한 함수) ───────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> str:
        """
        [기능] 최종 답변 생성 (라우팅 로직)
        
        query: 사용자 질문
        context: RAG 검색에서 찾은 참고 문서 (모드 A에서는 무시됨)
        mode: "A" | "B" | "C" (값을 안 주면 settings.py의 기본값을 따름)
        """
        
        # 1. 현재 작동할 모드 결정 (인자가 우선, 없으면 전역 설정)
        active_mode = mode or settings.model_mode

        # 2. 모드에 따른 컨텍스트(RAG) 필터링
        # 모드 A는 "LLM Only"이므로 검색된 문서가 있어도 무시(None)합니다.
        effective_context = context if active_mode in ("B", "C") else None
        
        # 3. 모델에게 던져줄 최종 텍스트 조립
        prompt = self._build_prompt(query, effective_context)

        logger.info(f"[GenerationService] 작동 시작 (Mode: {active_mode}, Backend: {settings.model_backend})")

        # 4-1. 백엔드가 OpenAI인 경우의 분기
        if settings.model_backend == "openai":
            client = self._get_openai()
            return client.request(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
            )

        # 4-2. 백엔드가 Qwen(로컬)인 경우의 분기
        if active_mode == "C":
            # LoRA 모드이면 어댑터가 결합된 클라이언트를 가져옵니다.
            client = self._get_qwen_lora()
        else:
            # 모드 A, B이면 순수 베이스 클라이언트를 가져옵니다.
            client = self._get_qwen_base()

        # 5. 선택된 클라이언트(뇌)에게 프롬프트를 던지고 답변을 받아옵니다.
        return client.request(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
        )