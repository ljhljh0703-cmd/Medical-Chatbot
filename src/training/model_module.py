"""
모델 로드 모듈.

역할:
  - Qwen 토크나이저 + 모델 로드 (finetune_lora.py 모델 로딩 흡수)
  - BitsAndBytes 4-bit 양자화 (선택)
  - LoRA 설정 적용 (PEFT)
  - LoRA 어댑터 병합 + 저장 (merge_adapter.py 흡수)
  - train_config.yaml dict 연동

기존 training/finetune_lora.py (모델 부분) + training/merge_adapter.py 로직 통합.
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from __future__ import annotations

from typing import Optional


# ─── 토크나이저 ──────────────────────────────────────────────────────────────
def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    """AutoTokenizer 로드. pad_token 미설정 시 eos_token으로 대체."""
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ─── 베이스 모델 로드 ────────────────────────────────────────────────────────
def load_base_model(
    model_name: str,
    torch_dtype: str = "float16",
    device_map: str = "auto",
    use_quantization: bool = True,
    trust_remote_code: bool = True,
):
    """
    베이스 모델 로드.

    use_quantization=True 이면 BitsAndBytes 4-bit 양자화 적용.
    VRAM 부족 환경(Colab T4)에서 권장.
    """
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM  # type: ignore

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    bnb_config = None
    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("[model_module] bitsandbytes 없음 — 양자화 비활성화")

    @staticmethod
    def apply_lora(model, train_cfg):
        """
        [기능]
        거대한 베이스 모델의 뇌(가중치)는 꽁꽁 얼려두고(Freeze), 
        학습이 가능한 아주 작은 메모지(LoRA 어댑터)만 모델 곳곳에 붙여주는 작업입니다.
        
        [예상 결과]
        전체 가중치(100%)를 학습하지 않고, 약 0.1% 미만의 가중치만 학습하게 되어 
        학습 속도가 엄청나게 빨라지고 GPU 과부하를 막습니다.
        """
        
        peft_config = LoraConfig(
            # r (Rank): 새로 붙이는 메모지의 '크기(차원)'. 
            # 클수록(32, 64) 똑똑해지지만 메모리를 많이 먹고, 작을수록(8, 16) 빠르지만 복잡한 논리를 못 배울 수 있습니다.
            r=train_cfg['lora_r'],
            
            # lora_alpha: 학습된 새로운 지식을 기존 뇌에 얼마나 강하게 주입할지 정하는 '증폭기(볼륨)'. 
            # 보통 r의 2배(r=16이면 alpha=32)로 설정하는 것이 국룰입니다.
            lora_alpha=train_cfg['lora_alpha'],
            
            # target_modules: 모델의 어느 부위에 이 메모지를 붙일 것인가?
            # q, k, v, o (어텐션: 문맥 파악 뇌) + gate, up, down (MLP: 추론 및 지식 저장 뇌)
            # 의학적 논리 추론(CoT)을 잘하려면 이 두 부위에 모두 붙이는 것이 좋습니다.
            target_modules=train_cfg['target_modules'],
            
            # lora_dropout: 학습할 때 실수로 메모지의 정보 일부(예: 5%)를 가려버리는 기술.
            # 모델이 특정 단어나 문장에 너무 의존하는 것(과적합, Overfitting)을 막아줍니다.
            lora_dropout=train_cfg['lora_dropout'],
            
            # bias="none": 기존 모델의 편향(Bias) 값은 건드리지 않겠다는 의미 (표준 설정)
            bias="none",
            
            # task_type: 우리가 하려는 작업의 종류 (LLM처럼 다음 단어를 예측하는 모델은 무조건 CAUSAL_LM)
            task_type="CAUSAL_LM"
        )
        
        # 설정한 LoRA(메모지)를 실제 베이스 모델(뇌)에 부착하여 최종 학습용 모델을 반환합니다.
        return get_peft_model(model, peft_config)

    @staticmethod
    def load_existing_lora_for_training(model, checkpoint_path):
        """
        [기능] 
        백지상태의 LoRA가 아닌, 이미 특정 스테이지(예: Stage 1)에서 
        학습이 완료된 LoRA 체크포인트를 불러와서 '이어서 학습'할 수 있도록 장착합니다.
        
        [핵심 파라미터]
        is_trainable=True : 이 옵션이 없으면 가중치가 읽기 전용(Read-only)으로 고정되어 
        Loss가 떨어지지 않고 에러가 발생합니다.
        """
        print(f"🔄 [Load] 기존 학습된 LoRA 가중치를 불러옵니다: {checkpoint_path}")
        model = PeftModel.from_pretrained(
            model, 
            checkpoint_path, 
            is_trainable=True  # 이어서 파인튜닝을 하기 위한 핵심 마스터키!
        )
        return model
