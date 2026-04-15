"""
Qwen 모델 추론 클라이언트 (PEFT 동적 로드 지원).

[주요 수정 내역 (Changelog)]
1. 파라미터 분리: 기존 단일 `model_path`에서 -> `base_model_path`(뼈대)와 `adapter_path`(LoRA 메모지)로 역할을 분리했습니다.
2. 동적 PEFT 로딩: 미리 병합된 14GB짜리 무거운 모델을 통째로 부르지 않고, 베이스 모델을 먼저 메모리에 올린 뒤 `peft` 라이브러리를 사용해 수십 MB짜리 LoRA 어댑터만 실시간으로 탈부착하도록 구조를 바꿨습니다.
3. VRAM 효율성 극대화: 이제 모드(Base <-> LoRA)를 전환해도 거대 모델이 메모리에 중복으로 올라가서 터지는(OOM) 현상이 완벽히 차단됩니다.
"""

from typing import Optional

class QwenClient:
    def __init__(self, base_model_path: str = "Qwen/Qwen2.5-7B-Instruct", adapter_path: Optional[str] = None):
        """
        [클래스 초기화]
        객체가 생성될 때 경로 이름만 변수에 저장하고, 실제 무거운 모델 다운로드나 로딩은 아직 하지 않습니다. (지연 로딩을 위한 준비)
        
        - base_model_path: 뼈대가 되는 원본 모델 경로 (예: 허깅페이스 주소 또는 로컬 폴더)
        - adapter_path: 학습이 완료된 LoRA 가중치가 있는 폴더 경로 (없으면 베이스 모델로만 동작)
        """
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        
        # 모델과 토크나이저를 담을 빈 그릇을 준비합니다.
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """
        [핵심 모델 로딩 로직 (Lazy Loading)]
        실제 첫 번째 질문(request)이 들어왔을 때 딱 한 번만 실행되어 모델을 VRAM(GPU)에 적재합니다.
        """
        
        # 💡 방어 코드: 이미 모델이 VRAM에 올라와 있다면(None이 아니라면), 
        # 아래 무거운 로딩 과정을 쳐다보지도 않고 즉시 통과(return)합니다. 추론 속도를 ms 단위로 만들어주는 핵심입니다.
        if self._model is not None:
            return
            
        try:
            # 사용할 때만 라이브러리를 메모리에 올립니다. (최적화)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            print(f"🔄 [Load] 토크나이저 및 베이스 모델 로딩 중: {self.base_model_path}")
            
            # 1. 토크나이저(단어 사전) 로드
            # trust_remote_code=True: Qwen 모델 특유의 커스텀 파이썬 코드를 실행하도록 허용합니다.
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path, trust_remote_code=True
            )
            
            # 2. 베이스 모델(뼈대 뇌) 로드
            # 여기서 약 14GB의 VRAM이 사용됩니다.
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,  # 16비트(절반 정밀도)로 로드하여 VRAM 사용량을 50% 절약합니다.
                device_map="auto",          # 빈 GPU 메모리를 알아서 찾아 모델을 최적으로 쪼개서 올려줍니다.
                trust_remote_code=True,
            )

            # 3. LoRA 어댑터(새로운 지식 메모지) 동적 부착
            if self.adapter_path:
                print(f"✨ [Load] 베이스 모델 위에 LoRA 어댑터 부착 중: {self.adapter_path}")
                from peft import PeftModel
                
                # 💡 핵심 마법: VRAM에 올라간 base_model 객체를 그대로 둔 채, 
                # adapter_path 폴더 안의 가중치만 가져와서 지정된 부위에 찰싹 붙입니다. (VRAM 추가 소모 거의 없음)
                self._model = PeftModel.from_pretrained(base_model, self.adapter_path)
            else:
                # 어댑터 경로가 없다면(모드 A, B), 그냥 순수 베이스 모델을 최종 모델로 사용합니다.
                self._model = base_model
                
            print("✅ 로딩 완료!")
            
        except ImportError:
            # 라이브러리가 안 깔려있을 때 친절하게 안내합니다.
            raise ImportError(
                "[QwenClient] transformers 또는 peft 패키지가 필요합니다. (pip install transformers peft)"
            )

    def request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        [실제 텍스트 생성 (추론) 로직]
        사용자의 질문을 받아, 토큰으로 쪼개고, 모델에 넣고, 나온 답변 토큰을 다시 사람의 언어로 번역합니다.
        """
        
        # 1. 로딩 확인
        # 모델이 안 켜져있으면 켜고, 켜져있으면 0.1초만에 바로 통과합니다.
        self._load()

        # 2. 대화 기록(Messages) 조립
        # OpenAI API 형식처럼 역할(role)과 내용(content)을 딕셔너리로 묶습니다.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt}) # "너는 의사다" 같은 시스템 설정
        messages.append({"role": "user", "content": prompt})              # 실제 사용자의 질문

        # 3. 채팅 템플릿 적용 (Chat Template)
        # Qwen 모델이 좋아하는 형식(예: <|im_start|>user\n질문<|im_end|>)으로 텍스트를 자동 포맷팅합니다.
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 4. 텍스트 -> 숫자(텐서) 변환
        # 포맷팅된 텍스트를 숫자로 바꾸고, 모델이 올라가 있는 GPU 메모리(device)로 데이터를 쏴줍니다.
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        import torch
        # 5. 추론 실행 (기억 봉인)
        # torch.no_grad(): "지금은 학습(파인튜닝)할 때가 아니니까, 기울기(기억) 계산은 하지 마!" 라고 선언하여 메모리와 속도를 극대화합니다.
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,                       # 입력 숫자들
                max_new_tokens=max_new_tokens,  # 최대 몇 글자(토큰)까지 대답할 것인가
                temperature=temperature,        # 0에 가까울수록 진지하고 일관되게, 1에 가까울수록 창의적이고 다양하게 대답
                do_sample=temperature > 0,      # temperature가 0보다 크면 확률적 뽑기(샘플링)를 활성화
            )

        # 6. 정답 추출 및 숫자 -> 텍스트 변환
        # output_ids에는 [질문 토큰 + 대답 토큰]이 같이 들어있습니다. 
        # 슬라이싱([inputs["input_ids"].shape[-1]:])을 통해 '질문' 부분은 잘라내고 순수 '대답' 토큰만 가져옵니다.
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        
        # 숫자를 다시 사람의 언어로 디코딩하고 반환합니다. (특수 토큰 <s> 등은 제거)
        return self._tokenizer.decode(generated, skip_special_tokens=True)