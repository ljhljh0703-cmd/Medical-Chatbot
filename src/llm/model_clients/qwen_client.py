"""
Qwen 모델 추론 클라이언트.

로컬 모델(병합된 가중치) 또는 HuggingFace Hub 모델을 로드하여 추론 수행.
베이스 모델 경로는 config/settings.py에서 주입받아 교체 가능하도록 설계.

사용 예시:
    client = QwenClient(model_path="Qwen/Qwen2.5-7B-Instruct")
    response = client.request("질문 내용")
"""

from typing import Optional


class QwenClient:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        model_path: HuggingFace 모델 ID 또는 로컬 경로.
        모델 교체 시 이 인자만 변경하면 됨.
        """
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """첫 호출 시 모델/토크나이저를 지연 로드(lazy loading)."""
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        except ImportError:
            raise ImportError(
                "[QwenClient] transformers 패키지가 필요합니다: pip install transformers"
            )

    def request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        prompt: 사용자 입력 또는 RAG 컨텍스트가 포함된 프롬프트
        system_prompt: 시스템 프롬프트 (None이면 config/prompts.py 기본값 사용)
        """
        self._load()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        import torch
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # 입력 토큰 제거 후 디코딩
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
