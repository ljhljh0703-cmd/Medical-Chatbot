"""
모델 로드 모듈.

역할:
  - Qwen 토크나이저 + 모델 로드 (finetune_lora.py 모델 로딩 흡수)
  - BitsAndBytes 4-bit 양자화 (선택): 메모리 사용량 1/4 절감
  - LoRA 설정 적용 (PEFT): 효율적인 파라미터 미세 조정
  - LoRA 어댑터 병합 + 저장 (merge_adapter.py 흡수)
  - train_config.yaml dict 연동

기존 training/finetune_lora.py (모델 부분) + training/merge_adapter.py 로직 통합.
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

class MedicalModelModule:
    @staticmethod
    def load_base_model(model_id, quant_cfg):
        """
        [역할] 거대 모델(Base Model)을 4-bit로 압축하여 로드합니다.
        [사용 기법] QLoRA (Quantized LoRA) - VRAM 24GB 이하에서도 7B 모델 학습 가능.
        """
        # compute_dtype: 계산할 때의 정밀도. bfloat16은 AI 전용 가속 데이터 타입입니다.
        compute_dtype = torch.float16 if quant_cfg.get('bnb_4bit_compute_dtype') == "float16" else torch.bfloat16

        # BitsAndBytesConfig: 모델의 '가중치'를 압축하는 상세 설정입니다.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get('load_in_4bit', True),
            bnb_4bit_quant_type=quant_cfg.get('bnb_4bit_quant_type', "nf4"), # nf4: 압축 효율이 가장 좋은 방식
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get('bnb_4bit_use_double_quant', True) # 메타데이터까지 압축
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",             # GPU/CPU 메모리에 적절히 분산 배치
            attn_implementation="sdpa",    # 최신 고속 어텐션 연산기 사용
            torch_dtype=compute_dtype,
            trust_remote_code=True
        )
        
        # 학습 안정성을 위해 캐시를 끄고, k-bit 학습용 전처리를 수행합니다.
        model.config.use_cache = False
        return prepare_model_for_kbit_training(model)

    @staticmethod
    def apply_lora(model, train_cfg):
        """
        [역할] 기존 모델은 고정(Freeze)하고, '학습 가능한 메모지(LoRA)'만 붙입니다.
        """
        peft_config = LoraConfig(
            # r (Rank): 메모지의 크기. 클수록 똑똑해지지만 메모리를 더 사용합니다.
            r=train_cfg.get('lora_r', 16),
            # lora_alpha: 새로운 지식을 기존 지식에 얼마나 강하게 섞을지 정하는 증폭률입니다.
            lora_alpha=train_cfg.get('lora_alpha', 32),
            # target_modules: 모델의 어느 부위를 학습시킬지 정합니다. (Qwen 최적화 레이어들)
            target_modules=train_cfg.get('target_modules', 
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=train_cfg.get('lora_dropout', 0.05),
            bias="none",
            task_type="CAUSAL_LM"
        )
        return get_peft_model(model, peft_config)

    @staticmethod
    def load_existing_lora_for_training(model, checkpoint_path):
        """[역할] 이전 스테이지에서 학습된 가중치를 불러와 '이어서' 학습합니다."""
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
        return model
