"""
모델 로드 모듈.

역할:
  - Qwen 토크나이저 + 모델 로드 (finetune_lora.py 모델 로딩 흡수)
  - LoRA 설정 적용 (PEFT)
  - train_config.yaml dict 연동

기존 training/finetune_lora.py (모델 부분) + training/merge_adapter.py 로직 통합.
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class MedicalModelModule:
    """
    [클래스 개요] 
    거대한 LLM(예: Qwen 7B)을 일반적인 GPU(VRAM 24GB 이하)에서도 
    메모리 터짐(OOM) 없이 원활하게 학습시킬 수 있도록 모델을 압축하고 준비하는 핵심 모듈입니다.
    """

    @staticmethod
    def load_base_model(model_id, quant_cfg):
        """
        [기능] 
        원본 AI 모델(Base Model)을 4-bit로 극단적으로 압축(양자화)하여 메모리에 올립니다.
        
        [예상 결과]
        원래 로딩 시 약 28GB가 필요한 7B 모델이, 이 함수를 거치면 약 5~6GB 수준으로 다이어트하여 로드됩니다.
        """
        
        # 1. 연산 정밀도 설정 (YAML에서 읽어옴)
        # 컴퓨터가 소수점을 계산할 때 사용할 방식을 정합니다. bfloat16은 AI 학습에 특화되어 속도가 빠릅니다.
        compute_dtype = torch.float16 if quant_cfg.get('bnb_4bit_compute_dtype') == "float16" else torch.bfloat16

        # 2. 양자화(Quantization) 설정
        # 모델의 '가중치(뇌 세포)'를 16비트에서 4비트로 찌그러뜨려 용량을 줄이는 마법입니다.
        bnb_config = BitsAndBytesConfig(
            # load_in_4bit: 모델을 4비트 크기로 메모리에 올릴지 여부 (True = 메모리 대폭 절약)
            load_in_4bit=quant_cfg.get('load_in_4bit', True),
            
            # bnb_4bit_quant_type: 숫자를 압축하는 방식 ('nf4'는 정규분포를 활용해 4비트 압축의 정보 손실을 최소화하는 최신 기법)
            bnb_4bit_quant_type=quant_cfg.get('bnb_4bit_quant_type', "nf4"),
            
            # bnb_4bit_compute_dtype: 저장만 4비트로 하고, 실제 계산할 때는 16비트로 풀어서 계산하도록 지정 (성능 유지)
            bnb_4bit_compute_dtype=compute_dtype,
            
            # bnb_4bit_use_double_quant: 압축 과정에서 생기는 '메타데이터'조차 한 번 더 압축할지 여부 (VRAM 추가 절약)
            bnb_4bit_use_double_quant=quant_cfg.get('bnb_4bit_use_double_quant', True)
        )

        # 3. 베이스 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            
            # device_map="auto": GPU가 여러 대이거나 GPU/CPU 메모리가 섞여 있을 때, 알아서 최적의 위치에 쪼개서 올려줌
            device_map="auto",
            
            # attn_implementation="sdpa": PyTorch 최신 버전에 있는 '초고속 어텐션 계산기'를 켜서 학습 속도를 1.5배 이상 올림
            attn_implementation="sdpa",
            torch_dtype=compute_dtype
        )
        
        # 4. 4비트 학습을 위한 최종 준비 (필수 안전장치)
        # 기울기 체크포인팅(Gradient Checkpointing)을 활성화하여, 
        # 학습 중 지나간 계산 값을 메모리에서 지우고 필요할 때 다시 계산하도록 만들어 VRAM을 극적으로 아낍니다.
        return prepare_model_for_kbit_training(model)

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