# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import load_dataset
from torch.utils.data import Subset
import re
import json


### 新增自定义数据集处理
# English dataset

class Step1Dataset:

    def __init__(self):
        pass

    def get_train_data(self):
        with open("/mnt/workspace/project/llm/local_proj/DeepSpeed-Chat/data/step1/train.json", "r") as fr:
            data = json.load(fr)
        return data

    def get_eval_data(self):
        with open("/mnt/workspace/project/llm/local_proj/DeepSpeed-Chat/data/step1/valid.json", "r") as fr:
            data = json.load(fr)
        return data

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


class Step2Dataset:

    def __init__(self):
        pass

    def get_train_data(self):
        with open("/mnt/workspace/project/llm/local_proj/DeepSpeed-Chat/data/step2/train.json", "r") as fr:
            data = json.load(fr)
        return data

    def get_eval_data(self):
        with open("/mnt/workspace/project/llm/local_proj/DeepSpeed-Chat/data/step2/valid.json", "r") as fr:
            data = json.load(fr)
        return data

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


class Step3Dataset:

    def __init__(self):
        pass

    def get_train_data(self):
        with open("/mnt/workspace/project/llm/local_proj/DeepSpeed-Chat/data/step3/train.json", "r") as fr:
            data = json.load(fr)
        return data

    def get_eval_data(self):
        with open("/mnt/workspace/project/llm/local_proj/DeepSpeed-Chat/data/step3/valid.json", "r") as fr:
            data = json.load(fr)
        return data

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']