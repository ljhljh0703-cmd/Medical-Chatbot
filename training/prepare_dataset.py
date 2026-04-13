"""
라벨링 데이터를 LoRA Fine-Tuning용 Alpaca 포맷으로 변환하고 train/test 분할.

입력 JSON 스키마:
{
    "qa_id": int,
    "domain": int,
    "q_type": int,
    "question": str,
    "answer": str,
    "source_folder": str (optional),
    "source_file": str (optional)
}

출력 Alpaca 포맷:
{
    "instruction": str,   ← question
    "input": "",
    "output": str,        ← answer
    "category": str,      ← domain 코드 → 이름 매핑
    "source_id": str      ← source_file (없으면 qa_id)
}
"""

import json
import random
import os


DOMAIN_MAP = {17: "내과"}

def convert_to_alpaca(item: dict) -> dict:
    category = DOMAIN_MAP.get(item.get("domain"), str(item.get("domain", "")))
    source_id = item.get("source_file") or str(item.get("qa_id", ""))
    return {
        "instruction": item["question"],
        "input": "",
        "output": item["answer"],
        "category": category,
        "source_id": source_id,
    }


def prepare(
    input_path: str,
    output_dir: str,
    test_size: int = 200,
    seed: int = 42,
) -> None:
    """
    input_path: 라벨링 데이터 JSON 파일 경로
    output_dir: train.json / test.json 저장 디렉터리
    test_size: 테스트셋 샘플 수 (초반 테스트 기준 200개)
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
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"[prepare_dataset] train: {len(train_data)}개 / test: {len(test_data)}개 저장 완료")
    print(f"  → {output_dir}/train.json, test.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="라벨링 데이터 JSON 경로")
    parser.add_argument("--output_dir", default="data/processed", help="출력 디렉터리")
    parser.add_argument("--test_size", type=int, default=200)
    args = parser.parse_args()
    prepare(args.input, args.output_dir, args.test_size)
