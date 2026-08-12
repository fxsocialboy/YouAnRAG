#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="artifacts/stage7/after_fix"
mkdir -p "$OUT/logs" artifacts/qdrant_local

if [[ ! -f artifacts/qdrant_local/meta.json ]]; then
  python scripts/import_stage1_to_qdrant.py --recreate | tee "$OUT/logs/00_import_qdrant.log"
fi
python scripts/preflight_stage74_autodl.py --out "$OUT/preflight.json" | tee "$OUT/logs/00_preflight.log"

python evaluate_stage7.py --dataset regression --backend v2 \
  --device cuda --reranker-device cuda --batch-size 64 \
  --embedding-model-path models/bge-large-zh-v1.5 \
  --reranker-model-path models/bge-reranker-base \
  --composer-mode deepseek --hyde-mode deepseek \
  --out "$OUT/regression_eval.json" --retry-errors \
  | tee "$OUT/logs/01_regression_calibration.log"

python scripts/validate_stage74_calibration.py \
  --input "$OUT/regression_eval.json" --out "$OUT/calibration_acceptance.json"

tar -czf artifacts/stage74_calibration_results.tar.gz \
  "$OUT/regression_eval.json" "$OUT/calibration_acceptance.json" "$OUT/preflight.json" "$OUT/logs"
echo "[OK] Stage7.4 calibration passed"
echo "Next: bash scripts/run_stage7_autodl.sh"
