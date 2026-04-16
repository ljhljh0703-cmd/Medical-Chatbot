"""
의료용 챗봇 메인 학습 스크립트 (3단계 커리큘럼 러닝)

역할:
  - 설정 파일(YAML) 로드 및 환경 세팅
  - 4-bit 양자화 모델 로드 및 LoRA 설정 (YAML 연동)
  - 일반 SFT(전체 텍스트) 데이터셋 전처리
  - 단계별(Stage 1, 2, 3) 학습 실행 및 메모리 관리

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
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

# 프로젝트 루트 경로를 인식시켜 src 폴더 안의 모듈을 불러올 수 있게 합니다.
sys.path.append(os.getcwd())

from src.training.data_module import MedicalDataModule
from src.training.model_module import MedicalModelModule

def main():
    # ------------------------------------------------------------
    # 1. 설정 및 환경 준비 (Setup & Config)
    # ------------------------------------------------------------
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(config['model']['id'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------
    # 2. 모델 및 데이터 모듈 초기화 (Initialization)
    # ------------------------------------------------------------
    model_module = MedicalModelModule()
    data_module = MedicalDataModule(tokenizer)

    model = model_module.load_base_model(config['model']['id'], config['quantization'])
    
    # [YAML 연동] lora_r, lora_alpha, target_modules 등이 config['training']을 통해 전달됩니다.
    model = model_module.apply_lora(model, config['training'])

    # ------------------------------------------------------------
    # 3. 커리큘럼 학습 루프 (Training Stages)
    # ------------------------------------------------------------
    for stage, params in config['training']['stages'].items():
        print(f"\n" + "="*60)
        print(f" [STAGE START] : {stage.upper()}")
        print(f" 길이={params['max_len']}, 배치={params['batch']}, 학습률={params['lr']}")
        print("="*60)

        train_ds = data_module.get_formatted_dataset(config['data'][stage])

        # ------------------------------------------------------------
        # [YAML 연동] 하드코딩 제거 및 Config 값 매핑
        # ------------------------------------------------------------
        sft_config = SFTConfig(
            output_dir=os.path.join(config['model']['output_dir'], stage),
            per_device_train_batch_size=params['batch'],
            learning_rate=params['lr'],
            num_train_epochs=params['epochs'],
            
            # YAML에서 가져온 설정들
            max_length=params['max_len'], 
            gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 8),
            lr_scheduler_type=config['training'].get('lr_scheduler_type', 'cosine'),
            warmup_ratio=config['training'].get('warmup_ratio', 0.05),
            
            # 일반 SFT 텍스트 필드 지정 (CoT 콜레이터 미사용)
            dataset_text_field="text", 
            
            # 하드웨어 최적화 고정 설정
            bf16=True,
            optim="paged_adamw_32bit",
            gradient_checkpointing=True,
            report_to="none"
        )

        trainer = SFTTrainer(
            model=model, 
            args=sft_config, 
            train_dataset=train_ds
            # data_collator 제거됨 (일반 SFT 학습)
        )

        trainer.train()

        # ------------------------------------------------------------
        # 4. 스테이지 종료 후 메모리 청소 (Memory Cleanup)
        # ------------------------------------------------------------
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 5. 최종 결과물 저장 (Final Save)
    # ------------------------------------------------------------
    model.save_pretrained(config['model']['final_dir'])
    tokenizer.save_pretrained(config['model']['final_dir'])
    print(f"\n 모든 학습 완료! 최종 모델 경로: {config['model']['final_dir']}")

if __name__ == "__main__":
    main()