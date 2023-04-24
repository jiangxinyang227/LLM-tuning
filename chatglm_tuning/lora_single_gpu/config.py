class Config:
    epochs = 4
    log_every = 10
    eval_every = 500
    checkpoint_every = 500

    train_steps = 1500
    warmup_steps = 1

    batch_size = 1
    accu_steps = 128
    max_input_len = 96
    max_output_len = 32
    learning_rate = 1e-3
    weight_decay = 0

    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules = ["query_key_value"]

    val_set_size = 2000
    data_path = "../../data/trans_chinese_alpaca_data.json"
    base_model = "THUDM/chatglm-6b"
    lora_model = "./lora_model"