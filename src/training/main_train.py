"""
의료용 챗봇 메인 학습 스크립트 (3단계 커리큘럼 러닝)

역할:
  - 설정 파일(YAML) 로드 및 환경 세팅
  - 4-bit 양자화 모델 로드 및 LoRA 설정
  - CoT(사고의 사슬) 대응 데이터셋 전처리
  - 단계별(Stage 1, 2, 3) 학습 실행 및 메모리 관리
  - valid set을 통한 최적 모델 학습

실행 방법:

    # 프로젝트 루트에서
    python src/training/main_train.py --config configs/train_config.yaml

    # run_train.sh 를 통해
    bash run_train.sh
"""

import os
import sys
import yaml
import torch
import gc
from transformers import AutoTokenizer, EarlyStoppingCallback # [수정] EarlyStoppingCallback 추가
from trl import SFTTrainer, SFTConfig

# 프로젝트 루트 경로를 인식시켜 src 폴더 안의 모듈을 불러올 수 있게 합니다.
sys.path.append(os.getcwd())

from src.training.data_module import MedicalDataModule
from src.training.model_module import MedicalModelModule

def main():
    # ------------------------------------------------------------
    # 1. 설정 및 환경 준비 (Setup & Config)
    # ------------------------------------------------------------
    # YAML 파일을 읽어 하이퍼파라미터(학습률, 배치 사이즈 등)를 가져옵니다.
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 토크나이저(단어 사전)를 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(config['model']['id'])
    # Qwen 등 일부 모델은 문장의 끝을 알리는 토큰을 패딩(빈칸 채우기) 토큰으로 공용합니다.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------
    # 2. 모델 및 데이터 모듈 초기화 (Initialization)
    # ------------------------------------------------------------
    model_module = MedicalModelModule()
    data_module = MedicalDataModule(tokenizer)

    # 베이스 모델을 4-bit로 압축해서 불러옵니다 (VRAM 절약의 핵심).
    model = model_module.load_base_model(config['model']['id'], config['quantization'])
    
    # 모델에 LoRA 어댑터(학습 가능한 작은 메모지)를 붙입니다.
    model = model_module.apply_lora(model, config['training'])

    # [수정] 고정 검증 데이터셋 로드 (데이터 오염 방지 및 일관된 평가용)
    print(f"🔍 [Load] 고정 검증 데이터셋 로드: {config['data']['valid']}")
    eval_ds = data_module.get_formatted_dataset(config['data']['valid'])

    # ------------------------------------------------------------
    # 3. CoT 학습을 위한 특수 설정 (Data Collator)
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # 4. 커리큘럼 학습 루프 (Training Stages)
    # ------------------------------------------------------------
    # 설정파일에 정의된 Stage 1(단답), 2(추론), 3(판별)을 순서대로 실행합니다.
    for stage, params in config['training']['stages'].items():
        print(f"\n" + "="*60)
        print(f"[STAGE START] : {stage.upper()}")
        print(f"전략: 길이={params['max_len']}, 배치={params['batch']}, 학습률={params['lr']}")
        print("="*60)

        # 현재 스테이지에 맞는 데이터셋 로드 및 포맷팅
        train_ds = data_module.get_formatted_dataset(config['data'][stage])

        # 학습 세부 설정 (SFTConfig)
        sft_config = SFTConfig(
            output_dir=os.path.join(config['model']['output_dir'], stage),
            per_device_train_batch_size=params['batch'], # 한 번에 GPU에 올릴 데이터 개수
            gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 8), # "할부 결제" 배치를 키우는 효과
            learning_rate=params['lr'],      # 공부하는 속도
            num_train_epochs=params['epochs'], # 전체 데이터를 몇 번 반복해서 볼 것인가
            max_length=params['max_len'],     # CoT 대응을 위해 문장 길이를 넉넉히 잡음
            dataset_text_field="text",
            
            # [수정] 최적화 및 검증 전략 (Loss 반등 방지)
            eval_strategy="steps",            # 에포크가 아닌 스텝 단위 평가
            eval_steps=100,                   # 100스텝마다 검증 수행
            save_strategy="steps",            # 평가 시점에 맞춰 체크포인트 저장
            save_steps=100,
            load_best_model_at_end=True,      # 학습 종료 후 가장 성능 좋았던 모델 로드
            metric_for_best_model="eval_loss",# 최적 모델 판단 기준
            greater_is_better=False,          # Loss는 낮을수록 좋음
            save_total_limit=2,               # 가장 좋은 모델 등 2개만 유지하여 용량 아낌
            
            bf16=True,                       # 최신 GPU 가속 활성화
            optim="paged_adamw_8bit",        # [RAM 최적화] VRAM/RAM 절약을 위해 8bit 옵티마이저 사용
            gradient_checkpointing=True,     # VRAM을 아끼기 위해 중간 계산값을 필요할 때 다시 계산
            report_to="none"
        )

        # 학습 실행기(Trainer) 생성 및 가동
        trainer = SFTTrainer(
            model=model, 
            args=sft_config, 
            train_dataset=train_ds,
            eval_dataset=eval_ds,             # [수정] 검증 데이터셋 추가
            # [수정] 3회 이상 성능 개선 없을 시 조기 종료하는 감시관 추가
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()

        # ------------------------------------------------------------
        # 5. 스테이지 종료 후 메모리 청소 (Memory Cleanup)
        # ------------------------------------------------------------
        # 스테이지가 끝날 때마다 찌꺼기 메모리를 비워줘야 다음 스테이지에서 안 터집니다.
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 6. 최종 결과물 저장 (Final Save)
    # ------------------------------------------------------------
    # 모든 스테이지를 통과한 최종 LoRA 가중치를 저장합니다.
    model.save_pretrained(config['model']['final_dir'])
    tokenizer.save_pretrained(config['model']['final_dir'])
    print(f"\n 모든 학습 완료! 최종 모델 경로: {config['model']['final_dir']}")

if __name__ == "__main__":
    main()