import torch
from torch import nn
from chatglm import ChatGLMForConditionalGeneration
from peft import LoraConfig, get_peft_model
from config import Config


class ChatGLMLoraModel(nn.Module):
    def __init__(self):
        super(ChatGLMLoraModel, self).__init__()

        model = ChatGLMForConditionalGeneration.from_pretrained(
            Config.base_model,
            torch_dtype=torch.float16)

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

    def forward(self, input_ids, labels):
        output = self.peft_model(
            input_ids=input_ids,
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