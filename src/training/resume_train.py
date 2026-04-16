"""
이어서 학습하기(Resume) 및 LoRA 파라미터 커스텀 스크립트

역할:
  - 기존 가중치 로드 또는 새로운 LoRA 차원(Rank) 설정
  - 학습률 및 LoRA 하이퍼파라미터 유연한 변경
  - CoT 데이터셋을 활용한 추가 미세 조정(Fine-Tuning)
"""

import os
import sys
import yaml
import torch
import gc
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

sys.path.append(os.getcwd())

from src.training.data_module import MedicalDataModule
from src.training.model_module import MedicalModelModule

def resume_training(base_checkpoint_path, new_data_path, output_dir, 
                    epochs=2, lr=0.0001, lora_r=16, lora_alpha=32):
    # ------------------------------------------------------------
    # 1. 환경 설정 로드 (Setup)
    # ------------------------------------------------------------
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(config['model']['id'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_module = MedicalModelModule()
    # 베이스 모델 로드
    model = model_module.load_base_model(config['model']['id'], config['quantization'])
    
    # ------------------------------------------------------------
    # 2. 모델 상태 결정 (Load existing vs New LoRA)
    # ------------------------------------------------------------
    if base_checkpoint_path:
        # [주의] 기존 가중치를 불러올 때는 기존에 저장된 r, alpha 값이 강제 적용됩니다.
        print(f" [Load] 기존 가중치를 이어받습니다: {base_checkpoint_path}")
        print(f" 기존 체크포인트 로드 시 lora_r({lora_r}) 설정은 무시되고 기존 설정이 유지됩니다.")
        model = model_module.load_existing_lora_for_training(model, base_checkpoint_path)
    else:
        # [신규] 베이스 모델에 내가 원하는 차원(r)으로 새로 붙일 때
        print(f"새로운 설정을 적용합니다: lora_r={lora_r}, lora_alpha={lora_alpha}")
        
        # 가변 파라미터를 적용하기 위해 임시로 설정 dict 수정
        custom_train_cfg = config['training'].copy()
        custom_train_cfg['lora_r'] = lora_r
        custom_train_cfg['lora_alpha'] = lora_alpha
        
        model = model_module.apply_lora(model, custom_train_cfg)

    # ------------------------------------------------------------
    # 3. 데이터 준비 및 CoT 전용 설정 (Data Preparation)
    # ------------------------------------------------------------
    data_module = MedicalDataModule(tokenizer)
    train_ds = data_module.get_formatted_dataset(new_data_path)
    
    # ------------------------------------------------------------
    # 4. 학습 세부 파라미터 설정 (Hyperparameters)
    # ------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_length=1024,           # CoT의 긴 추론 과정을 담기 위한 충분한 길이
        dataset_text_field="text",
        bf16=True,                 # Ampere 아키텍처 이상 GPU 가속
        gradient_checkpointing=True,
        optim="paged_adamw_32bit", # VRAM 부족 시 시스템 RAM 활용
        report_to="none"
    )

    # ------------------------------------------------------------
    # 5. 학습 엔진 가동 및 저장 (Execution)
    # ------------------------------------------------------------
    trainer = SFTTrainer(
        model=model, 
        args=sft_config, 
        train_dataset=train_ds,
    )
    
    print(f" 학습 시작... (데이터: {new_data_path})")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 메모리 자원 반납 (런타임 터짐 방지)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f" 학습 완료 및 저장 성공: {output_dir}")

if __name__ == "__main__":
    # ------------------------------------------------------------
    # [사용자 제어판] 여기서 모든 설정을 조절하세요!
    # ------------------------------------------------------------
    
    # 1. 베이스가 될 가중치 경로 (없으면 None)
    PREVIOUS_WEIGHTS = None 
    
    # 2. 학습할 데이터셋 경로
    NEW_DATASET = "data/processed/merged_3000_sampled.jsonl"
    
    # 3. 결과물이 저장될 폴더 이름
    SAVE_PATH = "src/training/lora_custom_rank_16"
    
    # 4. LoRA 핵심 차원 설정 (r이 커질수록 모델이 복잡한 패턴을 잘 배웁니다)
    # 
    MY_LORA_R = 16      # 보통 8, 16, 32, 64 중에서 선택
    MY_LORA_ALPHA = 32  # 보통 r의 2배로 설정하는 것이 국룰입니다.
    
    # 5. 학습 강도 조절
    CURRENT_LR = 1e-4   # 학습률 (Learning Rate)
    CURRENT_EPOCHS = 3  # 반복 횟수

    resume_training(
        base_checkpoint_path=PREVIOUS_WEIGHTS,
        new_data_path=NEW_DATASET,
        output_dir=SAVE_PATH,
        epochs=CURRENT_EPOCHS,
        lr=CURRENT_LR,
        lora_r=MY_LORA_R,
        lora_alpha=MY_LORA_ALPHA
    )
  