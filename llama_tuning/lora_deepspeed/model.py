import torch
from torch import nn
from transformers import LlamaForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from config import Config


class LlamaLoraModel(nn.Module):
    def __init__(self):
        super(LlamaLoraModel, self).__init__()
        model = LlamaForCausalLM.from_pretrained(
            Config.base_model,
            load_in_8bit=Config.load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto"  # 设置为auto时会默认使用所有可以使用的gpu，并且将模型分片加载。
        )  # 权重类型是float16

        # if Config.load_in_8bit:
        #     model = prepare_model_for_int8_training(model)

        lora_config = LoraConfig(
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            target_modules=Config.lora_target_modules,
            lora_dropout=Config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.peft_model = get_peft_model(model, lora_config)
        self.peft_model.config.use_cache = False

    def forward(self, input_ids, attention_mask, labels):
        output = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        loss = output["loss"]
        logits = output["logits"]
        predictions = logits.argmax(dim=-1)

        return loss, predictions

    def print_trainable_parameters(self):
        self.peft_model.print_trainable_parameters()

    def save_lora_model(self, lora_model):
        self.peft_model.save_pretrained(lora_model)