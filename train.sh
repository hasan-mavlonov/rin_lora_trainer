#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE=1
STEPS=1500
LEARNING_RATE=1e-4
MODEL="stabilityai/stable-diffusion-xl-base-1.0"
TRAIN_DIR="images"
OUTPUT_DIR="/output"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --learning-rate) LEARNING_RATE="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --train-dir) TRAIN_DIR="$2"; shift 2;;
    --output) OUTPUT_DIR="$2"; shift 2;;
    *) shift;;
  esac
done

accelerate launch scripts/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path "${MODEL}" \
  --train_data_dir "${TRAIN_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution 1024 \
  --train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps 4 \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler cosine \
  --lr_warmup_steps 0 \
  --max_train_steps "${STEPS}" \
  --checkpointing_steps 300 \
  --seed 42 \
  --rank 16 \
  --train_text_encoder \
  --mixed_precision fp16
