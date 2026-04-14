"""
[레거시 shim] 이 파일은 하위 호환성을 위해 유지됩니다.
실제 구현은 src/training/model_module.merge_and_save() 에 있습니다.

새 코드에서는 아래를 사용하세요:
    from training.model_module import merge_and_save
    merge_and_save(base_model, adapter_path, output_dir)
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

from training.model_module import merge_and_save  # noqa: E402


def merge(base_model: str, adapter_path: str, output_dir: str) -> None:
    """하위 호환 래퍼. src/training/model_module.merge_and_save() 으로 위임."""
    merge_and_save(base_model, adapter_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[shim] src/training/model_module 위임")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", default="models/qwen-lora-medical")
    parser.add_argument("--output_dir", default="models/qwen-merged-medical")
    args = parser.parse_args()
    merge(args.base_model, args.adapter_path, args.output_dir)
