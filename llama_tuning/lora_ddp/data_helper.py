import json
import random

import torch
from torch.utils.data import Dataset

from config import Config


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    batch = list(zip(*batch))
    input_ids = torch.tensor(batch[0], dtype=torch.long)
    attention_mask = torch.tensor(batch[1], dtype=torch.float16)
    labels = torch.tensor(batch[2], dtype=torch.long)

    return input_ids, attention_mask, labels


class DataHelper:
    def __init__(self):
        self.data_path = Config.data_path
        self.val_set_size = Config.val_set_size

    def load_data(self):
        with open(self.data_path, "r") as fr:
            data = json.load(fr)

        return data

    def gen_data(self):
        data = self.load_data()
        random.shuffle(data)

        train_data = data[self.val_set_size:]
        valid_data = data[:self.val_set_size]
        return train_data, valid_data


class LlamaDataset(Dataset):

    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.padding_side = "left"
        self.data = data
        self.sequence_len = Config.sequence_len
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.unk_token_id  # =0
        self.label_pad_token_id = -100  # pytorch 中label默认为-100时不会计算loss

    def generate_prompt(self, data_point):
        # sorry about the formatting disaster gotta move fast
        if data_point["input"]:
            return f"""给定任务的描述和输入的问题，请返回结果。
    描述:
    {data_point["instruction"]}
    输入:
    {data_point["input"]}
    回答:
    {data_point["output"]}"""
        else:
            return f"""给定问题的描述，请返回结果。
    描述:
    {data_point["instruction"]}
    回答:
    {data_point["output"]}"""

    def tokenize(self, prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.sequence_len,
            padding=False,
            return_tensors=None
        )
        input_ids, attention_mask, labels = [], [], []
        if (
                result["input_ids"][-1] != self.eos_token_id
                and len(result["input_ids"]) < self.sequence_len
                and add_eos_token
        ):
            result["input_ids"].append(self.eos_token_id)
            result["attention_mask"].append(1)

        pad_len = self.sequence_len - len(result["input_ids"])
        if pad_len <= 0:
            input_ids = result["input_ids"][:self.sequence_len]
            attention_mask = result["attention_mask"][:self.sequence_len]
            labels = input_ids.copy()
        else:
            input_ids = [self.pad_token_id] * pad_len + result["input_ids"]
            attention_mask = [0] * pad_len + result["attention_mask"]
            labels = [self.label_pad_token_id] * pad_len + result["input_ids"]

        return input_ids, attention_mask, labels

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.generate_prompt(data_point)
        input_ids, attention_mask, labels = self.tokenize(full_prompt)
        return input_ids, attention_mask, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        input_ids, attention_mask, labels = self.generate_and_tokenize_prompt(data_point)

        return (input_ids, attention_mask, labels)