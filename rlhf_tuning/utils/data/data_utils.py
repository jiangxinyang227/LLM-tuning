# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0


import json
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from itertools import chain
from . import raw_datasets


def get_raw_dataset(train_phase):
    train_data_path, valid_data_path = "", ""
    if train_phase == 1:
        train_data_path = "../data/step1/train.json"
        valid_data_path = "../data/step1/valid.json"
    elif train_phase == 2:
        train_data_path = "../data/step2/train.json"
        valid_data_path = "../data/step2/valid.json"
    else:
        train_data_path = "../data/step3/train.json"
        valid_data_path = "../data/step3/valid.json"

    with open(train_data_path, "r") as fr:
        train_data = json.load(fr)

    with open(valid_data_path, "r") as fr:
        valid_data = json.load(fr)

    return train_data, valid_data


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"], self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id


def create_dataset(data, train_phase, tokenizer, end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(data):
            # tokenize the text
            chosen_sentence = tmp_data["prompt"] + tmp_data["chosen"]
            chosen_sentence += end_of_conversation_token
            chosen_token = tokenizer(chosen_sentence,
                                     max_length=max_seq_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                0)
            chosen_token["attention_mask"] = chosen_token[
                "attention_mask"].squeeze(0)
            chosen_dataset.append(chosen_token)

    elif train_phase == 2:
        for i, tmp_data in enumerate(data):
            # tokenize the text
            chosen_sentence = tmp_data["prompt"] + tmp_data["chosen"]
            reject_sentence = tmp_data["prompt"] + tmp_data["rejected"]
            chosen_sentence += end_of_conversation_token  # the accept response
            reject_sentence += end_of_conversation_token
            chosen_token = tokenizer(chosen_sentence,
                                     max_length=max_seq_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            reject_token = tokenizer(reject_sentence,
                                     max_length=max_seq_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            chosen_token["input_ids"] = chosen_token["input_ids"]
            chosen_token["attention_mask"] = chosen_token["attention_mask"]
            chosen_dataset.append(chosen_token)

            reject_token["input_ids"] = reject_token["input_ids"]
            reject_token["attention_mask"] = reject_token["attention_mask"]
            reject_dataset.append(reject_token)

    elif train_phase == 3:
        for i, tmp_data in enumerate(data):
            # tokenize the text
            prompt = tmp_data["prompt"]
            prompt_token = tokenizer(prompt, return_tensors="pt")
            prompt_token["input_ids"] = prompt_token["input_ids"]
            prompt_token["attention_mask"] = prompt_token["attention_mask"]
            for key_word in ["input_ids", "attention_mask"]:
                length = prompt_token[key_word].size()[-1]
                if length > max_seq_len:
                    y = prompt_token[key_word].squeeze(0)[length -
                                                          (max_seq_len -
                                                           1):].flip(0)
                else:
                    y = prompt_token[key_word].squeeze(0).flip(0)
                prompt_token[key_word] = y
            prompt_dataset.append(prompt_token)
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_prompt_dataset_custom(local_rank,
                                 train_phase,
                                 tokenizer,
                                 max_seq_len,
                                 end_of_conversation_token="<|endoftext|>"):
    # if local_rank <= 0:
    train_data, valid_data = get_raw_dataset(train_phase)

    print("train data num: ", len(train_data))
    print("valid data num: ", len(valid_data))

    train_dataset = create_dataset(train_data, train_phase, tokenizer, end_of_conversation_token, max_seq_len)
    valid_dataset = create_dataset(valid_data, train_phase, tokenizer, end_of_conversation_token, max_seq_len)

    return train_dataset, valid_dataset


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(pad_length, 0),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(pad_length, 0),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                       self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []