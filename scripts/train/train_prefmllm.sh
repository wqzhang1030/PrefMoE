#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./checkpoints/vicuna-7b-v1.5}"
VISION_TOWER="${VISION_TOWER:-./checkpoints/clip-vit-large-patch14-336}"
MM_PROJECTOR="${MM_PROJECTOR:-./checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin}"
DATA_PATH="${DATA_PATH:-./data/mmpb_clean/sample.csv}"
SPLIT_PATH="${SPLIT_PATH:-./data/mmpb_clean/split.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./data}"
PSEUDO_USER_CSV="${PSEUDO_USER_CSV:-./data/mmpb_clean/pseudo_users.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/prefmllm_hmoe}"
NUM_PSEUDO_USERS="${NUM_PSEUDO_USERS:-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PYTHON="${PYTHON:-python3}"

"${PYTHON}" -m llava.train.train_prefmoe \
  --train_mode prefmllm \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --version vicuna_v1 \
  --data_path "${DATA_PATH}" \
  --data_split_path "${SPLIT_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --vision_tower "${VISION_TOWER}" \
  --pretrain_mm_mlp_adapter "${MM_PROJECTOR}" \
  --mm_projector_type mlp2x_gelu \
  --image_aspect_ratio pad \
  --injection_description_prompt_type hard_moderate \
  --injection_preference_prompt_type explicit \
  --identity_mode name \
  --name_memory_pseudo_csv_path "${PSEUDO_USER_CSV}" \
  --name_memory_num_pseudo_users "${NUM_PSEUDO_USERS}" \
  --name_memory_module2_arch hierarchical_moe_lora \
  --name_memory_task_router_mode query \
  --name_memory_task_router_supervision none \
  --name_memory_task_router_loss_weight 0.0 \
  --name_memory_pref_router_mode learned \
  --name_memory_pref_router_loss_weight 0.0 \
  --name_memory_profile_router_mode learned \
  --name_memory_enable_counterfactual True \
  --name_memory_use_factorized_preference_memory True \
  --name_memory_pref_residual_loss_mode density_focal \
  --name_memory_pref_density_eta 1.0 \
  --name_memory_loss_weight_consistency 0.05 \
  --name_memory_loss_weight_profile_img 0.10 \
  --name_memory_loss_weight_description 0.10 \
  --name_memory_loss_weight_pref_factor 0.20 \
  --name_memory_loss_weight_pref_decorrelation 0.05 \
  --name_memory_loss_weight_pref_focal 0.20 \
  --name_memory_contrastive_temperature 0.07 \
  --name_memory_focal_gamma 2.0 \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --model_max_length 8192 \
  --bf16 False \
  --fp16 True \
  --logging_steps 1 \
  --save_steps 1000 \
  --save_total_limit 1
