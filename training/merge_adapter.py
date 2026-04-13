"""
LoRA 어댑터를 베이스 모델에 병합하는 유틸리티.

학습 후 어댑터(adapter_model.bin)를 베이스 모델에 합쳐
단일 모델 가중치 파일로 저장.

실행 예시:
    python merge_adapter.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --adapter_path models/qwen-lora-medical \
        --output_dir models/qwen-merged-medical
"""

import argparse


def merge(base_model: str, adapter_path: str, output_dir: str) -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    print(f"[merge_adapter] 베이스 모델 로드: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"[merge_adapter] 어댑터 로드: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[merge_adapter] 병합 완료. 저장 경로: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", default="models/qwen-lora-medical")
    parser.add_argument("--output_dir", default="models/qwen-merged-medical")
    args = parser.parse_args()
    merge(args.base_model, args.adapter_path, args.output_dir)
