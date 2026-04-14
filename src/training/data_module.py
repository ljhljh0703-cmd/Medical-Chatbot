"""
학습 데이터 모듈.

역할:
  - 라벨링 JSON/JSONL 로드 (Alpaca / CoT 포맷)
  - Alpaca 포맷 변환 (prepare_dataset.py 로직 통합)
  - train/test 분할
  - 프롬프트 포맷팅
  - DataCollatorForCompletionOnlyLM 마스킹 (Response 부분만 loss 계산)

기존 training/prepare_dataset.py 로직을 완전히 흡수.
"""

from __future__ import annotations

import json
import os
import random
from typing import Literal


# ─── 포맷 템플릿 ────────────────────────────────────────────────────────────
ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

COT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Chain of Thought:\n{chain_of_thought}\n\n"
    "### Response:\n{output}"
)

DOMAIN_MAP: dict[int, str] = {17: "내과"}


# ─── Alpaca 변환 (prepare_dataset.py 흡수) ──────────────────────────────────
def convert_to_alpaca(item: dict) -> dict:
    """QAPair JSON → Alpaca 포맷 변환."""
    category = DOMAIN_MAP.get(item.get("domain"), str(item.get("domain", "")))
    source_id = item.get("source_file") or str(item.get("qa_id", ""))
    return {
        "instruction": item["question"],
        "input": "",
        "output": item["answer"],
        "category": category,
        "source_id": source_id,
    }


def prepare_and_split(
    input_path: str,
    output_dir: str,
    test_size: int = 200,
    seed: int = 42,
) -> tuple[str, str]:
    """
    라벨링 JSON → Alpaca 변환 후 train/test 분할 저장.

    Returns:
        (train_path, test_path)
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    converted = [convert_to_alpaca(item) for item in data]
    random.seed(seed)
    random.shuffle(converted)

    test_data = converted[:test_size]
    train_data = converted[test_size:]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.json")
    test_path = os.path.join(output_dir, "test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"[data_module] train: {len(train_data)}개 / test: {len(test_data)}개 저장 완료")
    print(f"  → {train_path}, {test_path}")
    return train_path, test_path


# ─── 포맷팅 ─────────────────────────────────────────────────────────────────
def format_example(
    example: dict,
    fmt: Literal["alpaca", "cot"] = "alpaca",
) -> dict:
    """단일 샘플을 학습용 text 필드로 포맷."""
    if fmt == "cot":
        text = COT_TEMPLATE.format(
            instruction=example.get("instruction", ""),
            input=example.get("input", ""),
            chain_of_thought=example.get("chain_of_thought", ""),
            output=example.get("output", ""),
        )
    else:
        text = ALPACA_TEMPLATE.format(
            instruction=example.get("instruction", ""),
            input=example.get("input", ""),
            output=example.get("output", ""),
        )
    return {"text": text}


# ─── HuggingFace Dataset 로드 ────────────────────────────────────────────────
def load_dataset_from_file(
    path: str,
    fmt: Literal["alpaca", "cot", "jsonl"] = "alpaca",
):
    """JSON 또는 JSONL 파일 → HuggingFace Dataset (text 필드 포함)."""
    from datasets import Dataset  # type: ignore

    ext = os.path.splitext(path)[-1].lower()
    if ext == ".jsonl" or fmt == "jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if isinstance(records, dict):
            records = [records]

    effective_fmt: Literal["alpaca", "cot"] = "cot" if fmt == "cot" else "alpaca"
    dataset = Dataset.from_list(records)
    dataset = dataset.map(lambda ex: format_example(ex, fmt=effective_fmt))
    return dataset


# ─── DataCollator ────────────────────────────────────────────────────────────
def get_data_collator(tokenizer, model=None):
    """
    DataCollatorForCompletionOnlyLM — '### Response:\\n' 이후만 loss 계산.
    trl 미설치 시 DataCollatorForSeq2Seq로 fallback.
    """
    try:
        from trl import DataCollatorForCompletionOnlyLM  # type: ignore

        response_template = "### Response:\n"
        return DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )
    except (ImportError, Exception):
        from transformers import DataCollatorForSeq2Seq  # type: ignore

        return DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
