#!bin/bash


deepspeed --include localhost:1 --master_port=12345 main.py \
    --model_name_or_path bigscience/bloom-560m \
    --disable_dropout \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --max_seq_len 64 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 3 \
    --deepspeed \
    --offload \
    --output_dir ./output