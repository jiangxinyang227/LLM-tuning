import json
import random

import torch
from torch.utils.data import Dataset

from config import Config


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    batch = list(zip(*batch))
    input_ids = torch.tensor(batch[0], dtype=torch.long)
    labels = torch.tensor(batch[1], dtype=torch.long)

    return input_ids, labels


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


class ChatGLMDataset(Dataset):

    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data
        self.max_input_len = Config.max_input_len
        self.max_output_len = Config.max_output_len
        self.label_pad_token_id = -100  # pytorch 中label默认为-100时不会计算loss

    def generate_prompt(self, data_point):
        # sorry about the formatting disaster gotta move fast
        if data_point["input"]:
            prompt = "{}: {}".format(data_point["instruction"], data_point["input"])
        else:
            prompt = "{}".format(data_point["instruction"])

        output = data_point["output"]
        return prompt, output

    def tokenize(self, prompt, output):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = self.tokenizer.encode(text=output, add_special_tokens=False)

        if len(a_ids) > self.max_input_len - 1:
            a_ids = a_ids[: self.max_input_len - 1]

        if len(b_ids) > self.max_output_len - 2:
            b_ids = b_ids[: self.max_output_len - 2]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

        context_length = input_ids.index(self.tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [self.label_pad_token_id] * context_length + input_ids[mask_position + 1:]

        pad_len = (self.max_input_len + self.max_output_len) - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.label_pad_token_id] * pad_len

        return input_ids, labels

    def generate_and_tokenize_prompt(self, data_point):
        prompt, output = self.generate_prompt(data_point)
        input_ids, labels = self.tokenize(prompt, output)
        return input_ids, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        input_ids, labels = self.generate_and_tokenize_prompt(data_point)

        return (input_ids, labels)