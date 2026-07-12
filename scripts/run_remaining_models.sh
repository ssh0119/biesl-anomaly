#!/bin/bash
set -e
cd "$(dirname "$0")/.."

run() {
  local stage=$1 modality=$2
  echo "=== RUNNING stage=${stage} modality=${modality} ==="
  uv run python -m src.train --stage "$stage" --modality "$modality" --epochs 3 --batch-size 512 --num-workers 16 --lr 1e-3
  echo "=== DONE stage=${stage} modality=${modality} ==="
}

run 1 ppg
run 1 both
run 2 ecg
run 2 ppg
run 2 both

echo "ALL_RUNS_COMPLETE"
