"""
Qwen2.5-7B LoRA Fine-Tuning 스크립트 (Colab 실행 대응).

필요 패키지: transformers, peft, trl, datasets, torch, bitsandbytes
Colab Pro(A100 40GB) 기준으로 설정됨.
베이스 모델은 settings에서 변경 가능 (7B → 14B 등).

실행 예시:
    python finetune_lora.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --train_data data/processed/train.json \
        --output_dir models/qwen-lora-medical
"""

import argparse
import json


def load_dataset(path: str):
    """Alpaca 포맷 JSON → HuggingFace Dataset 변환."""
    from datasets import Dataset
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def format_prompt(example: dict) -> dict:
    """Alpaca 포맷 → 학습용 텍스트 변환."""
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}


def train(model_name: str, train_data_path: str, output_dir: str) -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    import torch

    print(f"[finetune_lora] 베이스 모델: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA 설정 — 향후 r, alpha 등 조정 가능
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Qwen 기준
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(train_data_path)
    dataset = dataset.map(format_prompt)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[finetune_lora] 학습 완료. 저장 경로: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_data", default="data/processed/train.json")
    parser.add_argument("--output_dir", default="models/qwen-lora-medical")
    args = parser.parse_args()
    train(args.model_name, args.train_data, args.output_dir)
