#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 내과 의료 챗봇 — LoRA 학습 파이프라인 원클릭 실행 스크립트
#
# 사용법:
#   bash run_train.sh
#   bash run_train.sh --config configs/train_config.yaml
#   bash run_train.sh --train_path data/processed/train.json
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ── 기본 인자 ──────────────────────────────────────────────────────────────
CONFIG="configs/train_config.yaml"
TRAIN_PATH="data/processed/train.json"
EVAL_PATH="data/processed/test.json"

# ── CLI 인자 파싱 ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2";     shift 2 ;;
        --train_path)  TRAIN_PATH="$2"; shift 2 ;;
        --eval_path)   EVAL_PATH="$2";  shift 2 ;;
        *) echo "알 수 없는 인자: $1"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════════"
echo "  내과 의료 챗봇 LoRA 학습 시작"
echo "  설정 파일  : $CONFIG"
echo "  학습 데이터: $TRAIN_PATH"
echo "  평가 데이터: $EVAL_PATH"
echo "═══════════════════════════════════════════════════"

# ── 1. 패키지 설치 확인 ────────────────────────────────────────────────────
echo "[1/3] 패키지 설치 확인..."
pip install -q -r requirements.txt

# ── 2. 학습 데이터 존재 확인 ───────────────────────────────────────────────
if [[ ! -f "$TRAIN_PATH" ]]; then
    echo "[2/3] ❌ 학습 데이터 없음 — prepare_dataset.py 를 먼저 실행하세요."
    echo "  예) python training/prepare_dataset.py --input data/raw/<파일>.json --output_dir data/processed"
    exit 1
fi
echo "[2/3] ✅ 학습 데이터 확인: $TRAIN_PATH"

# ── 3. 학습 실행 ─────────────────────────────────────────────────────────
echo "[3/3] 학습 시작..."
python src/training/main_train.py \
    --config "$CONFIG" \
    --train_path "$TRAIN_PATH" \
    --eval_path "$EVAL_PATH"

echo "═══════════════════════════════════════════════════"
echo "  ✅ 학습 완료"
echo "═══════════════════════════════════════════════════"
