#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="artifacts/stage7/after_fix"
LOG="$OUT/logs"
mkdir -p "$LOG" artifacts/qdrant_local

EMBED_MODEL="models/bge-large-zh-v1.5"
RERANK_MODEL="models/bge-reranker-base"

if [[ ! -f artifacts/qdrant_local/meta.json ]]; then
  python scripts/import_stage1_to_qdrant.py --recreate | tee "$LOG/00_import_qdrant.log"
fi

python scripts/preflight_stage74_autodl.py --out "$OUT/preflight.json" | tee "$LOG/00_preflight.log"
python scripts/validate_stage7_eval_sets.py | tee "$LOG/00_validate_datasets.log"

if [[ ! -f "$OUT/calibration_acceptance.json" ]] || \
   ! python -c 'import json,sys; sys.exit(0 if json.load(open("artifacts/stage7/after_fix/calibration_acceptance.json"))["passed"] else 1)'; then
  echo "[ERROR] calibration has not passed; run: bash scripts/run_stage74_calibration_autodl.sh" >&2
  exit 4
fi

python evaluate_stage7.py --dataset labeled --backend legacy \
  --device cuda --batch-size 64 --embedding-model-path "$EMBED_MODEL" \
  --out "$OUT/final_labeled_legacy_eval.json" --retry-errors | tee "$LOG/01_labeled_legacy.log"

python evaluate_stage7.py --dataset labeled --backend v2 \
  --device cuda --reranker-device cuda --batch-size 64 \
  --embedding-model-path "$EMBED_MODEL" --reranker-model-path "$RERANK_MODEL" \
  --composer-mode deepseek --hyde-mode deepseek \
  --out "$OUT/final_labeled_v2_eval.json" --retry-errors | tee "$LOG/02_labeled_v2.log"

python evaluate_stage7.py --dataset random --backend v2 \
  --device cuda --reranker-device cuda --batch-size 64 \
  --embedding-model-path "$EMBED_MODEL" --reranker-model-path "$RERANK_MODEL" \
  --composer-mode deepseek --hyde-mode deepseek \
  --out "$OUT/final_random_v2_eval.json" --retry-errors | tee "$LOG/03_random_v2.log"

python scripts/generate_final_report.py \
  --legacy-labeled "$OUT/final_labeled_legacy_eval.json" \
  --v2-labeled "$OUT/final_labeled_v2_eval.json" \
  --v2-random "$OUT/final_random_v2_eval.json" \
  --before-metrics artifacts/stage7/before_fix/final_metrics.json \
  --out "$OUT/final_report.md" --summary-out "$OUT/final_metrics.json"

python scripts/check_legacy_hash.py --out "$OUT/legacy_hash_report.json" || true

python scripts/smoke_youan_agent_stage7.py --real-llm --limit 3 \
  --device cuda --reranker-device cuda \
  --out "$OUT/youan_langgraph_real_smoke.json" | tee "$LOG/04_youan_langgraph_smoke.log"

python scripts/generate_stage7_manual_review.py \
  --labeled "$OUT/final_labeled_v2_eval.json" --random "$OUT/final_random_v2_eval.json" \
  --out "$OUT/stage7_manual_review.md" --count 15

python - <<'PY'
import hashlib, json, pathlib, subprocess
root=pathlib.Path.cwd(); out=root/'artifacts/stage7/after_fix'
files=[p for p in out.rglob('*') if p.is_file() and p.name!='package_manifest.json']
manifest={'git_commit':None,'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
try: manifest['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
except Exception: pass
(out/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY

tar -czf artifacts/stage74_after_fix_results.tar.gz \
  artifacts/stage7/after_fix experiments/eval_queries_final_labeled.jsonl \
  experiments/eval_queries_final_random.jsonl experiments/stage74_fix_regression.jsonl

echo "[OK] Stage7.4 after-fix evaluation finished"
echo "Download: $ROOT/artifacts/stage74_after_fix_results.tar.gz"
