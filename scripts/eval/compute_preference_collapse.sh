#!/usr/bin/env bash
set -euo pipefail

INPUT_CSV="${INPUT_CSV:-./outputs/eval_0turn/raw/fold_0/mt_00_et_00_raw.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_0turn/fpout}"
PYTHON="${PYTHON:-python3}"

"${PYTHON}" -m llava.eval.prefmllm.collapse_metrics \
  --input-csv "${INPUT_CSV}" \
  --output-dir "${OUTPUT_DIR}"
