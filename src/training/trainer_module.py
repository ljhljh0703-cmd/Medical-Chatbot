"""
SFTTrainer 설정 및 학습 루프 모듈.

역할:
  - TrainingArguments 구성 (train_config.yaml 연동)
  - SFTTrainer 생성 및 학습 실행
  - 체크포인트 저장

기존 training/finetune_lora.py 의 train() 함수 로직 통합.
"""

from __future__ import annotations

import os


def build_training_args(cfg: dict):
    """train_config.yaml training 섹션 → TrainingArguments."""
    from transformers import TrainingArguments  # type: ignore

    t = cfg.get("training", {})
    return TrainingArguments(
        output_dir=t.get("output_dir", "models/qwen-lora-medical"),
        num_train_epochs=t.get("num_train_epochs", 3),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        learning_rate=t.get("learning_rate", 2e-4),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.05),
        fp16=t.get("fp16", True),
        bf16=t.get("bf16", False),
        logging_steps=t.get("logging_steps", 50),
        save_strategy=t.get("save_strategy", "epoch"),
        save_total_limit=t.get("save_total_limit", 2),
        eval_strategy=t.get("evaluation_strategy", "epoch"),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        report_to=t.get("report_to", "none"),
        seed=t.get("seed", 42),
    )


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    cfg: dict,
    data_collator=None,
):
    """SFTTrainer 생성."""
    from trl import SFTTrainer  # type: ignore

    sft = cfg.get("sft", {})
    training_args = build_training_args(cfg)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field=sft.get("dataset_text_field", "text"),
        max_seq_length=sft.get("max_seq_length", 2048),
        packing=sft.get("packing", False),
        data_collator=data_collator,
    )
    return trainer


def run_training(trainer, output_dir: str) -> None:
    """학습 실행 + 어댑터 저장."""
    print(f"[trainer_module] 학습 시작")
    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    print(f"[trainer_module] 학습 완료. 어댑터 저장: {output_dir}")
