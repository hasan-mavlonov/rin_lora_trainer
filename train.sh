#!/bin/bash
accelerate launch scripts/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path ./models/sdxl-base \
  --train_data_dir ./images \
  --output_dir ./output \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 0 \
  --max_train_steps 1500 \
  --checkpointing_steps 300 \
  --seed 42 \
  --rank 16 \
  --train_text_encoder \
  --mixed_precision fp16
