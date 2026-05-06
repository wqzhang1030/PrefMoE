#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-./checkpoints/prefmllm/fold_0_task_0}"
MODEL_BASE="${MODEL_BASE:-./checkpoints/vicuna-7b-v1.5}"
DATA_PATH="${DATA_PATH:-./data/mmpb_clean/sample.csv}"
SPLIT_PATH="${SPLIT_PATH:-./data/mmpb_clean/split.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/eval_0turn}"
PYTHON="${PYTHON:-python3}"

"${PYTHON}" llava/eval/prefmoe/eval_demo/eval_prefmoe_bridge_2.py test \
  --model-path "${MODEL_PATH}" \
  --model-base "${MODEL_BASE}" \
  --fold 0 \
  --model-task 0 \
  --eval-task 0 \
  --data-path "${DATA_PATH}" \
  --data-split-path "${SPLIT_PATH}" \
  --image-folder "${IMAGE_FOLDER}" \
  --identity-mode name \
  --drop-profile-in-test \
  --name_memory_use_concept_id \
  --output-root "${OUTPUT_ROOT}" \
  --rank 0 \
  --world-size 1 \
  --max-samples -1

"${PYTHON}" llava/eval/prefmoe/eval_demo/eval_prefmoe_bridge_2.py evaluate \
  --model-base "${MODEL_BASE}" \
  --fold 0 \
  --model-task 0 \
  --eval-task 0 \
  --data-path "${DATA_PATH}" \
  --data-split-path "${SPLIT_PATH}" \
  --image-folder "${IMAGE_FOLDER}" \
  --identity-mode name \
  --drop-profile-in-test \
  --name_memory_use_concept_id \
  --output-root "${OUTPUT_ROOT}" \
  --rank 0 \
  --world-size 1
