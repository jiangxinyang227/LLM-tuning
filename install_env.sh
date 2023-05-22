#!bin/bash

#1, 安装pytorch，cuda版本是要大于11.6的
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# 2, 安装transformers，transformers版本要是最新的，不是的话卸载重新安装，否则不支持llama
pip install transformers
pip install sentence_transformers
pip install datasets

pip install peft
pip install accelerate
pip install deepspeed
pip install sentencepiece