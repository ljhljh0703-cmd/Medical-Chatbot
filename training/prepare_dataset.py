"""
[레거시 shim] 이 파일은 하위 호환성을 위해 유지됩니다.
실제 구현은 src/training/data_module.py 에 있습니다.

새 코드에서는 아래를 사용하세요:
    from training.data_module import prepare_and_split, convert_to_alpaca
"""

import os
import sys

# src/ 경로 추가
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_SRC = os.path.join(_ROOT, "src")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.data_module import (  # noqa: E402
    convert_to_alpaca,
    prepare_and_split,
    DOMAIN_MAP,
)


def prepare(
    input_path: str,
    output_dir: str,
    test_size: int = 200,
    seed: int = 42,
) -> None:
    """하위 호환 래퍼. src/training/data_module.prepare_and_split() 으로 위임."""
    prepare_and_split(input_path, output_dir, test_size, seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="[shim] src/training/data_module 위임")
    parser.add_argument("--input", required=True, help="라벨링 데이터 JSON 경로")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--test_size", type=int, default=200)
    args = parser.parse_args()
    prepare(args.input, args.output_dir, args.test_size)
