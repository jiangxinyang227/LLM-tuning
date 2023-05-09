import os
import gc
import time
import sys

import argparse

grandparent_path = os.path.abspath("..")
sys.path.insert(0, grandparent_path)

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from accelerate import Accelerator, DistributedType
from accelerate.utils import DummyOptim, DummyScheduler

from transformers import AdamW, get_linear_schedule_with_warmup, get_scheduler
from chatglm import ChatGLMTokenizer

from model import ChatGLMLoraModel
from data_helper import DataHelper, ChatGLMDataset, collate_fn
from metric import mean, accuracy
from utils import get_logger
from config import Config


def b2mb(x):
    return int(x / 2 ** 20)


# 追踪进程内存使用峰值的管理器
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *exc):
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


logger = get_logger("chatglm_lora", "log.txt")


class Trainer:
    def __init__(self):

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

        self.lora_model = Config.lora_model

        self.accelerator = Accelerator()
        print("dist type: ", self.accelerator.distributed_type)
        print("mix-precision: ", self.accelerator.mixed_precision)

        self.tokenizer = ChatGLMTokenizer.from_pretrained(Config.base_model)
        print("tokenizer init done: ", len(self.tokenizer))

        self.train_data_loader, self.valid_data_loader = self.get_data_loader()
        print("get data loader done")

        # 初始化模型对象
        self.model = ChatGLMLoraModel()
        print("model load done")

        self.optimizer, self.scheduler = self.create_optimizer_and_scheduler()

        self.model, self.optimizer, self.train_data_loader, self.valid_data_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_data_loader, self.valid_data_loader, self.scheduler
        )

        self.model.train()

    def get_data_loader(self):
        # 加载数据集
        data_obj = DataHelper()
        train_data, valid_data = data_obj.gen_data()

        logger.info("train data size: {}".format(len(train_data)))
        logger.info("valid data size: {}".format(len(valid_data)))

        train_data_set = ChatGLMDataset(self.tokenizer, train_data)
        valid_data_set = ChatGLMDataset(self.tokenizer, valid_data)

        train_data_loader = DataLoader(
            train_data_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn)
        valid_data_loader = DataLoader(
            valid_data_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn)

        return train_data_loader, valid_data_loader

    def create_optimizer_and_scheduler(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer_cls = (
            AdamW
            if self.accelerator.state.deepspeed_plugin is None
               or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
        )

        optimizer = optimizer_cls(optimizer_grouped_parameters, lr=self.learning_rate)

        if (
                self.accelerator.state.deepspeed_plugin is None
                or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.train_steps,
            )
        else:
            scheduler = DummyScheduler(
                optimizer, total_num_steps=self.train_steps, warmup_num_steps=self.warmup_steps
            )

        return optimizer, scheduler

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            eval_losses = []
            eval_word_preds = []
            eval_word_labels = []
            for batch_data in self.valid_data_loader:
                input_ids = batch_data[0].to(self.accelerator.device)
                labels = batch_data[1].to(self.accelerator.device)

                loss, predictions = self.model(input_ids, labels)

                loss, predictions, labels = self.accelerator.gather_for_metrics((loss, predictions, labels))
                loss = loss.mean()
                eval_losses.append(float(loss))
                eval_word_preds.extend(predictions.tolist())
                eval_word_labels.extend(labels.tolist())

            if self.accelerator.is_main_process:
                acc = accuracy(pred_ys=eval_word_preds, true_ys=eval_word_labels)

                logger.info("\n")
                logger.info("eval: num: {},  loss: {}, acc: {}".format(
                    len(eval_word_preds), mean(eval_losses), acc))
                logger.info("\n")

    def train(self):

        current_step = 1
        start = time.time()

        train_losses = []
        train_word_preds = []
        train_word_labels = []

        for epoch in range(self.epochs):
            logger.info("----- Epoch {}/{} -----".format(epoch + 1, self.epochs))

            for batch_data in self.train_data_loader:
                input_ids = batch_data[0].to(self.accelerator.device)
                labels = batch_data[1].to(self.accelerator.device)

                loss, predictions = self.model(input_ids, labels)

                # 为了训练加速，减少gpu间的通信，只在主进程上看训练的日志
                if self.accelerator.is_main_process:
                    train_losses.append(float(loss))
                    train_word_preds.extend(predictions.tolist())
                    train_word_labels.extend(labels.tolist())

                # 梯度累积训练
                loss /= self.accu_steps
                self.accelerator.backward(loss)

                if current_step % self.accu_steps == 0:
                    clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if current_step % (self.log_every * self.accu_steps) == 0:
                    if self.accelerator.is_main_process:
                        acc = accuracy(pred_ys=train_word_preds, true_ys=train_word_labels)
                        logger.info("train: step: {}, num: {}, loss: {}, acc: {}".format(
                            current_step // self.accu_steps, len(train_word_preds), mean(train_losses), acc))

                        train_losses = []
                        train_word_preds = []
                        train_word_labels = []

                if current_step % (self.eval_every * self.accu_steps) == 0:
                    self.eval()
                    self.accelerator.wait_for_everyone()
                    self.model.train()

                current_step += 1

                if (current_step // self.accu_steps) > self.train_steps:
                    break
            if (current_step // self.accu_steps) > self.train_steps:
                break

        end = time.time()
        print("total train time: ", end - start)

        # 保存模型
        self.accelerator.wait_for_everyone()
        self.model.module.peft_model.save_pretrained(
            self.lora_model,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model)
        )
        print("model save done")


def main():
    set_seed(0)

    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    main()