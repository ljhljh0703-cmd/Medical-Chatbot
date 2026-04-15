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

def resume_training(base_checkpoint_path, new_data_path, output_dir, epochs=2, lr=0.0001):
    """
    [기능] 
    1. base_checkpoint_path가 있으면 -> 기존 LoRA 가중치에서 이어서 학습
    2. base_checkpoint_path가 None이면 -> Qwen 초기 상태에서 새 LoRA를 만들어 학습
    """
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"\n" + "="*60)
    print(f"🌿 [EXPERIMENT BRANCH] 파생 학습 시작")
    print(f"👉 베이스 가중치: {base_checkpoint_path if base_checkpoint_path else 'Qwen Base (초기 상태)'}")
    print(f"👉 학습 데이터: {new_data_path}")
    print("="*60)

    # 1. 모델 준비
    tokenizer = AutoTokenizer.from_pretrained(config['model']['id'])
    model_module = MedicalModelModule()

    # 베이스 모델을 4bit로 로드
    model = model_module.load_base_model(config['model']['id'], config['quantization'])
    
    # 💡 [핵심 분기 로직] 체크포인트 유무에 따라 뇌의 상태를 결정!
    if base_checkpoint_path:
        # 과거의 기억(가중치)을 불러와서 이어서 학습
        print(f"🔄 [Load] 기존 학습된 LoRA 가중치를 이어받아 학습합니다.")
        model = model_module.load_existing_lora_for_training(model, base_checkpoint_path)
    else:
        # 백지상태의 Qwen에 새로운 메모지(LoRA)를 붙여서 처음부터 학습
        print("🆕 [New] Qwen 초기 상태에서 새로운 LoRA 어댑터를 생성하여 학습을 시작합니다.")
        model = model_module.apply_lora(model, config['training'])

    # 2. 데이터 준비
    data_module = MedicalDataModule(tokenizer)
    train_ds = data_module.get_formatted_dataset(new_data_path)

    # 3. 학습 설정
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_length=1024,
        dataset_text_field="text",
        completion_only_loss=True,
        bf16=True,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=50,
        report_to="none"
    )

    # 4. 학습 실행
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_ds)
    print("💡 파생 학습 로그:")
    trainer.train()

    # 5. 새로운 결과물 저장 및 메모리 정리
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n🎉 [COMPLETE] 파생 학습 완료! 결과물 경로: {output_dir}")

if __name__ == "__main__":
    # --- [사용자 제어판] ---
    
    # 💡 마법의 스위치: 
    # 기존 가중치 경로를 넣으면 이어서 학습하고, None으로 두면 쌩짜 Qwen에서 시작합니다.
    PREVIOUS_WEIGHTS = None  # 예시: "src/training/results/stage1" 또는 None
    
    NEW_DATASET = "data/processed/merged_하드코어_실험용.jsonl"
    
    SAVE_PATH = "src/training/lora_branch_from_base"
    
    # 백지상태에서 시작할 때는 학습률을 살짝 높게(0.0001) 주는 것이 좋습니다.
    # 이어서 학습할 때는 기존 지식이 망가지지 않게 살짝 낮게(0.00005) 줍니다.
    CURRENT_LR = 0.0001 if PREVIOUS_WEIGHTS is None else 0.00005

    resume_training(
        base_checkpoint_path=PREVIOUS_WEIGHTS,
        new_data_path=NEW_DATASET,
        output_dir=SAVE_PATH,
        epochs=3,
        lr=CURRENT_LR
    )