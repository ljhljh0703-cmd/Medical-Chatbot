"""
Colab 환경에서 학습을 실행하는 메인 엔트리포인트.

파이프라인:
  data_module → model_module → trainer_module

실행 방법:
    # 프로젝트 루트에서
    python src/training/main_train.py --config configs/train_config.yaml

    # run_train.sh 를 통해
    bash run_train.sh
"""

from __future__ import annotations

import argparse
import os
import sys

# 프로젝트 루트 / src 경로 추가 (Colab 환경 대비)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..")
_ROOT = os.path.join(_SRC, "..")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML이 필요합니다: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str, train_path: str | None, eval_path: str | None) -> None:
    cfg = _load_yaml(config_path)

    # CLI 인자 오버라이드
    if train_path:
        cfg.setdefault("data", {})["train_path"] = train_path
    if eval_path:
        cfg.setdefault("data", {})["eval_path"] = eval_path

    data_cfg = cfg.get("data", {})
    output_cfg = cfg.get("output", {})
    fmt = data_cfg.get("format", "alpaca")

    print(f"[main_train] 설정 파일  : {config_path}")
    print(f"[main_train] 학습 데이터: {data_cfg['train_path']}")
    print(f"[main_train] 평가 데이터: {data_cfg.get('eval_path', '없음')}")

    # ── 1. 데이터 로드 ──────────────────────────────────────────────────────
    from training.data_module import load_dataset_from_file, get_data_collator

    train_ds = load_dataset_from_file(data_cfg["train_path"], fmt=fmt)
    eval_ds = None
    eval_p = data_cfg.get("eval_path")
    if eval_p and os.path.exists(eval_p):
        eval_ds = load_dataset_from_file(eval_p, fmt=fmt)

    # ── 2. 모델 & 토크나이저 로드 ───────────────────────────────────────────
    from training.model_module import load_model_and_tokenizer_from_config

    model, tokenizer = load_model_and_tokenizer_from_config(cfg)
    collator = get_data_collator(tokenizer, model)

    # ── 3. 학습 ─────────────────────────────────────────────────────────────
    from training.trainer_module import build_trainer, run_training

    adapter_dir = output_cfg.get("adapter_dir", "models/qwen-lora-medical")
    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, cfg, data_collator=collator)
    run_training(trainer, adapter_dir)

    print(f"\n[main_train] 완료. 어댑터: {adapter_dir}")
    merged_dir = output_cfg.get("merged_dir")
    if merged_dir:
        print(f"[main_train] 병합이 필요하면: python src/training/main_train.py merge --adapter {adapter_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="내과 의료 챗봇 LoRA 학습")
    parser.add_argument("--config", default="configs/train_config.yaml", help="YAML 설정 파일 경로")
    parser.add_argument("--train_path", default=None, help="학습 데이터 경로 (YAML 오버라이드)")
    parser.add_argument("--eval_path", default=None, help="평가 데이터 경로 (YAML 오버라이드)")
    args = parser.parse_args()
    main(args.config, args.train_path, args.eval_path)
