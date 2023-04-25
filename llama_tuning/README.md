### llama-7b微调
    主要实现的是基于lora的微调
#### lora_single_gpu
    单gpu上微调，最小可以在16G的A100上微调。
#### lora_ddp
    多gpu的ddp微调，单个GPU必须得大于16G，否则无法存放下整个模型。
#### lora_deepspeed
    deepspeed zero-3 + cpu offload微调，可以在多张GPU显存更小卡上微调，
    经测试在2张A100上单张卡得显存最少只占用5G多一点。