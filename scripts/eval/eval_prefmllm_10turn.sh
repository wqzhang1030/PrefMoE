#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-./checkpoints/prefmllm/fold_0_task_0}"
MODEL_BASE="${MODEL_BASE:-./checkpoints/vicuna-7b-v1.5}"
DATA_PATH="${DATA_PATH:-./data/mmpb_clean/sample.csv}"
SPLIT_PATH="${SPLIT_PATH:-./data/mmpb_clean/split.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./data}"
GENERIC_CONVERSATION_PATH="${GENERIC_CONVERSATION_PATH:-./data/multi_turn/generic_text/multi_turn_conversation.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/eval_10turn}"
PYTHON="${PYTHON:-python3}"

"${PYTHON}" llava/eval/CVLMP/eval_demo/eval_cvlmp_bridge_2.py test \
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
  --generic-conversation-enable \
  --generic-conversation-n-turn 10 \
  --generic-conversation-path "${GENERIC_CONVERSATION_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --rank 0 \
  --world-size 1 \
  --max-samples -1

"${PYTHON}" llava/eval/CVLMP/eval_demo/eval_cvlmp_bridge_2.py evaluate \
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
  --generic-conversation-enable \
  --generic-conversation-n-turn 10 \
  --generic-conversation-path "${GENERIC_CONVERSATION_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --rank 0 \
  --world-size 1
