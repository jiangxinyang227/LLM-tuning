import random
import os
import json
import time
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup
from transformers import LlamaTokenizer
from peft import get_peft_model_state_dict

from model import LlamaLoraModel
from data_helper import DataHelper, LlamaDataset, collate_fn
from metric import mean, accuracy
from utils import get_logger
from config import Config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Trainer:
    def __init__(self):

        self.lora_model = Config.lora_model

        self.epochs = Config.epochs
        self.log_every = Config.log_every
        self.eval_every = Config.eval_every
        self.checkpoint_every = Config.checkpoint_every

        self.train_steps = Config.train_steps
        self.warmup_steps = Config.warmup_steps
        self.learning_rate = Config.learning_rate
        self.weight_decay = Config.weight_decay

        self.batch_size = Config.batch_size
        self.accu_steps = Config.accu_steps

        self.tokenizer = LlamaTokenizer.from_pretrained(Config.base_model)

        self.train_data_loader, self.valid_data_loader = self.get_data_loader()
        print("get data loader done")

        # 初始化模型对象
        self.model = LlamaLoraModel()
        self.model.peft_model.print_trainable_parameters()

        old_state_dict = self.model.state_dict
        self.model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(self.model, type(self.model))

        self.model = self.model.cuda()
        self.model.train()

        print("model load done")

        # for name, param in self.model.named_parameters():
        #     print(name, param.dtype)

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=self.optimizer,
                                                                   num_warmup_steps=self.warmup_steps,
                                                                   num_training_steps=self.train_steps,
                                                                   lr_end=0.0)
        self.scaler = GradScaler()  # 用于混合精度训练

    def get_data_loader(self):
        # 加载数据集
        data_obj = DataHelper()
        train_data, valid_data = data_obj.gen_data()

        logger.info("train data size: {}".format(len(train_data)))
        logger.info("valid data size: {}".format(len(valid_data)))

        train_data_set = LlamaDataset(self.tokenizer, train_data)
        valid_data_set = LlamaDataset(self.tokenizer, valid_data)

        train_data_loader = DataLoader(train_data_set, batch_size=self.batch_size, drop_last=True, num_workers=6,
                                       shuffle=True, collate_fn=collate_fn)
        valid_data_loader = DataLoader(valid_data_set, batch_size=self.batch_size, num_workers=6, collate_fn=collate_fn)

        return train_data_loader, valid_data_loader

    def train(self):

        current_step = 1
        start = time.time()

        train_losses = []
        train_word_preds = []
        train_word_labels = []
        train_masks = []

        for epoch in range(self.epochs):
            logger.info("----- Epoch {}/{} -----".format(epoch + 1, self.epochs))

            for batch_data in self.train_data_loader:
                input_ids = batch_data[0].cuda()
                attention_mask = batch_data[1].cuda()
                labels = batch_data[2].cuda()

                with autocast():
                    loss, predictions = self.model(input_ids, attention_mask, labels)

                train_losses.append(float(loss))
                train_word_preds.extend(predictions.tolist())
                train_word_labels.extend(labels.tolist())
                train_masks.extend(attention_mask.tolist())

                # 梯度累积训练
                loss /= self.accu_steps
                # loss.backward()

                # 放大loss，并求梯度
                self.scaler.scale(loss).backward()

                if current_step % self.accu_steps == 0:
                    # self.optimizer.step()
                    # 先将梯度缩放回去，再执行梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), 1.0)

                    self.scaler.step(self.optimizer)

                    # for name, param in self.model.named_parameters():
                    #     if param.requires_grad:
                    #         print(name, param.dtype, param.grad.dtype)

                    self.scheduler.step()
                    self.scaler.update()
                    self.optimizer.zero_grad()

                if current_step % (self.log_every * self.accu_steps) == 0:
                    acc = accuracy(pred_ys=train_word_preds, true_ys=train_word_labels, masks=train_masks)
                    logger.info("train: step: {}, loss: {}, acc: {}".format(
                        current_step // self.accu_steps, mean(train_losses), acc))

                    train_losses = []
                    train_word_preds = []
                    train_word_labels = []
                    train_masks = []

                if current_step % (self.eval_every * self.accu_steps) == 0:

                    self.model.eval()

                    with torch.no_grad():
                        eval_losses = []
                        eval_word_preds = []
                        eval_word_labels = []
                        eval_masks = []
                        for batch_data in self.valid_data_loader:
                            input_ids = batch_data[0].cuda()
                            attention_mask = batch_data[1].cuda()
                            labels = batch_data[2].cuda()

                            with autocast():
                                eval_loss, eval_predictions = self.model(input_ids, attention_mask, labels)

                            eval_losses.append(eval_loss)

                            eval_word_preds.extend(eval_predictions.tolist())
                            eval_word_labels.extend(labels.tolist())
                            eval_masks.extend(attention_mask.tolist())

                        acc = accuracy(pred_ys=eval_word_preds, true_ys=eval_word_labels, masks=eval_masks)

                        logger.info("\n")
                        logger.info("eval:  loss: {}, acc: {}".format(
                            mean(eval_losses), acc))
                        logger.info("\n")

                    self.model.train()

                if current_step % (self.checkpoint_every * self.accu_steps) == 0:
                    lora_model_path = self.lora_model + "/" + str(current_step // self.accu_steps)
                    if not os.path.exists(lora_model_path):
                        os.makedirs(lora_model_path)
                    self.model.save_lora_model(lora_model_path)

                current_step += 1

                if (current_step // self.accu_steps) > self.train_steps:
                    break
            if (current_step // self.accu_steps) > self.train_steps:
                break

        end = time.time()
        print("total train time: ", end - start)


if __name__ == "__main__":
    # 读取用户在命令行输入的信
    logger = get_logger("llama_lora", "log.txt")

    trainer = Trainer()
    trainer.train()