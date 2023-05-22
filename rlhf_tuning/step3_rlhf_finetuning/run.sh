#!bin/bash

CUDA_LAUNCH_BLOCKING=1

deepspeed --include localhost:0,1 main.py \
    --actor_model_name_or_path ../step1_supervised_finetuning/sft_model \
    --critic_model_name_or_path ../step2_reward_model_finetuning/rw_model \
    --per_device_train_batch_size 1 \
    --per_device_mini_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 1 \
    --max_prompt_seq_len 64 \
    --max_answer_seq_len 64 \
    --actor_zero_stage 3 \
    --critic_zero_stage 3 \
    --disable_actor_dropout \
    --deepspeed \
    --actor_lora_dim 64 \
    --actor_lora_module_name query_key_value \
    --critic_lora_dim 32 \
    --critic_lora_module_name query_key_value \
    --only_optimize_lora \
    --enable_hybrid_engine \
    --output_dir ./output