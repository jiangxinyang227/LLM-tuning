#!bin/bash

deepspeed --include localhost:0 main.py \
   --model_name_or_path /mnt/workspace/project/llm/models/bloom-1b1 \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 64 \
   --learning_rate 3e-5 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --output_dir ./output