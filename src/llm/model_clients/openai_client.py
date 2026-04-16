"""
OpenAI API 클라이언트.

Chat Completion 기반으로 답변 생성.
설정은 config/settings.py 에서 주입.
"""

from typing import Optional
from observability.logger import logger


class OpenAIClient:
    """
    OpenAI Chat Completion 래퍼.
    model_name, api_key 등은 외부에서 주입받는다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("[OpenAIClient] openai 패키지가 필요합니다.")
        return self._client

    def request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Chat Completion 요청.

        prompt: 사용자 메시지 (또는 RAG 컨텍스트 포함 프롬프트)
        system_prompt: 시스템 메시지 (None이면 생략)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[OpenAIClient] 요청 실패: {e}")
            return f"[OpenAI 오류] {e}"
