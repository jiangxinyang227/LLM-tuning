import random
import os
import json
import time
import sys
import functools
import argparse

grandparent_path = os.path.abspath("..")
sys.path.insert(0, grandparent_path)

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import BatchSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast

from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler

from transformers import AdamW, get_polynomial_decay_schedule_with_warmup
from chatglm import ChatGLMTokenizer

from model import ChatGLMLoraModel
from data_helper import DataHelper, ChatGLMDataset, collate_fn
from metric import mean, accuracy
from utils import get_logger
from config import Config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_distributed_mode(local_rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def reduce_value(value, average=True):
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= Config.world_size

        return value


def gather_value(value, gather_values):
    with torch.no_grad():
        dist.all_gather(gather_values, value)


class Trainer:
    def __init__(self, local_rank):
        self.local_rank = local_rank

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

        self.cpu_offload = Config.cpu_offload

        self.lora_model = Config.lora_model

        self.tokenizer = ChatGLMTokenizer.from_pretrained(Config.base_model)

        self.train_data_loader, self.valid_data_loader, self.train_sampler = self.get_data_loader()
        print("get data loader done")

        # 初始化模型对象
        model = ChatGLMLoraModel()
        print("model load done")
        self.model = model.to(self.local_rank)

        print("model load multi gpu done")

        optimizer_args = {"lr": self.learning_rate, "weight_decay": self.weight_decay}
        self.optimizer = OSS(params=model.parameters(), optim=AdamW, **optimizer_args)

        self.model = ShardedDDP(self.model, self.optimizer)
        self.model.train()

        self.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=self.optimizer,
                                                                   num_warmup_steps=self.warmup_steps,
                                                                   num_training_steps=self.train_steps,
                                                                   lr_end=0.0)
        self.scaler = ShardedGradScaler()

    def get_data_loader(self):
        # 加载数据集
        data_obj = DataHelper()
        train_data, valid_data = data_obj.gen_data()

        logger.info("train data size: {}".format(len(train_data)))
        logger.info("valid data size: {}".format(len(valid_data)))

        train_data_set = ChatGLMDataset(self.tokenizer, train_data)
        valid_data_set = ChatGLMDataset(self.tokenizer, valid_data)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data_set)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, self.batch_size, drop_last=True)
        train_data_loader = DataLoader(
            train_data_set,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn)
        valid_data_loader = DataLoader(
            valid_data_set,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
            sampler=valid_sampler,
            collate_fn=collate_fn)

        return train_data_loader, valid_data_loader, train_sampler

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            eval_losses = []
            eval_word_preds = []
            eval_word_labels = []
            for batch_data in self.valid_data_loader:
                input_ids = batch_data[0].cuda()
                labels = batch_data[1].cuda()

                with autocast():
                    loss, predictions = self.model(input_ids, labels)

                # 获取所有gpu上输出的数据
                avg_loss_multi_gpu = reduce_value(loss, average=True)
                gather_preds = [torch.zeros_like(predictions, dtype=predictions.dtype) for _ in
                                range(Config.world_size)]
                gather_labels = [torch.zeros_like(labels, dtype=labels.dtype) for _ in range(Config.world_size)]
                gather_value(predictions, gather_preds)
                gather_value(labels, gather_labels)

                eval_losses.append(float(avg_loss_multi_gpu))
                for pred, label in zip(gather_preds, gather_labels):
                    eval_word_preds.extend(pred.tolist())
                    eval_word_labels.extend(label.tolist())

            if is_main_process():
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
            self.train_sampler.set_epoch(epoch)

            for batch_data in self.train_data_loader:
                input_ids = batch_data[0].cuda()
                labels = batch_data[1].cuda()

                with autocast():
                    loss, predictions = self.model(input_ids, labels)

                # 为了训练加速，减少gpu间的通信，只在主进程上看训练的日志
                if is_main_process():
                    train_losses.append(float(loss))
                    train_word_preds.extend(predictions.tolist())
                    train_word_labels.extend(labels.tolist())

                # 梯度累积训练
                loss /= self.accu_steps
                self.scaler.scale(loss).backward()

                if current_step % self.accu_steps == 0:
                    # 先将梯度缩放回去，再执行梯度裁剪
                    self.scaler.unscale_(self.optimizer)

                    clip_grad_norm_(self.model.parameters(), 1.0)

                    self.scaler.step(self.optimizer)

                    self.scheduler.step()
                    self.scaler.update()
                    self.optimizer.zero_grad()

                if current_step % (self.log_every * self.accu_steps) == 0:
                    if is_main_process():
                        acc = accuracy(pred_ys=train_word_preds, true_ys=train_word_labels)
                        logger.info("train: step: {}, num: {}, loss: {}, acc: {}".format(
                            current_step // self.accu_steps, len(train_word_preds), mean(train_losses), acc))

                        train_losses = []
                        train_word_preds = []
                        train_word_labels = []

                if current_step % (self.eval_every * self.accu_steps) == 0:
                    self.eval()
                    dist.barrier()  # 等待所有进程跑完验证
                    self.model.train()

                if current_step % (self.checkpoint_every * self.accu_steps) == 0:
                    dist.barrier()
                    if is_main_process():
                        lora_model_path = self.lora_model + "/" + str(current_step // self.accu_steps)
                        if not os.path.exists(lora_model_path):
                            os.makedirs(lora_model_path)
                        self.model.module.save_lora_model(lora_model_path)

                current_step += 1

                if (current_step // self.accu_steps) > self.train_steps:
                    break
            if (current_step // self.accu_steps) > self.train_steps:
                break

        end = time.time()
        print("total train time: ", end - start)


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    local_rank = int(os.environ['LOCAL_RANK'])

    print("local_rank: ", local_rank)

    set_seed(0)

    # DDP backend初始化
    init_distributed_mode(local_rank, Config.world_size)
    print("初始化结束")

    logger = get_logger("chatglm_lora", "log.txt")

    trainer = Trainer(local_rank)
    trainer.train()
    cleanup()