"""
[레거시 shim] 이 파일은 하위 호환성을 위해 유지됩니다.
실제 구현은 src/training/ 에 있습니다.

  데이터 로드  : src/training/data_module.py
  모델 로드    : src/training/model_module.py
  학습 루프    : src/training/trainer_module.py
  메인 엔트리  : src/training/main_train.py

새 코드 / Colab 에서는 아래를 권장합니다:
    python src/training/main_train.py --config configs/train_config.yaml
"""

import argparse


import os
import sys

# src/ 경로 추가
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_SRC = os.path.join(_ROOT, "src")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def train(model_name: str, train_data_path: str, output_dir: str) -> None:
    """하위 호환 래퍼. src/training/ 모듈들로 위임."""
    from training.data_module import load_dataset_from_file, get_data_collator
    from training.model_module import load_tokenizer, load_base_model, apply_lora
    from training.trainer_module import build_trainer, run_training

    # 기본 설정값 (train_config.yaml 미사용 시 하드코딩 유지)
    cfg = {
        "training": {
            "output_dir": output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "fp16": True,
            "logging_steps": 50,
            "save_strategy": "epoch",
            "report_to": "none",
        },
        "sft": {"max_seq_length": 2048, "dataset_text_field": "text"},
    }

    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, use_quantization=False)
    model = apply_lora(model, target_modules=["q_proj", "v_proj"])

    dataset = load_dataset_from_file(train_data_path)
    collator = get_data_collator(tokenizer, model)

    trainer = build_trainer(model, tokenizer, dataset, None, cfg, data_collator=collator)
    run_training(trainer, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[shim] src/training/ 위임")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_data", default="data/processed/train.json")
    parser.add_argument("--output_dir", default="models/qwen-lora-medical")
    args = parser.parse_args()
    train(args.model_name, args.train_data, args.output_dir)
