class Config:
    epochs = 3
    log_every = 10
    eval_every = 500
    checkpoint_every = 500

    train_steps = 1500
    warmup_steps = 100

    batch_size = 8
    accu_steps = 8
    sequence_len = 200
    learning_rate = 3e-4
    weight_decay = 0

    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj"]

    val_set_size = 2000
    data_path = "../../data/trans_chinese_alpaca_data.json"
    base_model = "decapoda-research/llama-7b-hf"
    lora_model = "./lora_model"
    world_size = 2