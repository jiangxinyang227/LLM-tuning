### chatglm-7b微调
    主要实现的是基于lora的微调
#### lora_single_gpu
    单gpu上微调，最小可以在16G的A100上微调。
#### lora_ddp
    多gpu的ddp微调，单个GPU必须得大于16G，否则无法存放下整个模型。
#### lora_shared_dpp
    zero-2，多GPU上会对梯度和优化器分片，但由于本想是是lora微调，lora的参数量
    本来就很小，所以对内存的优化不明显
#### lora_deepspeed
    deepspeed 的 zero-3 + cpu offload微调，可以在多张GPU显存更小卡上微调，
    经测试在2张A100上单张卡得显存最少只占用5G多一点。
     deepspeed在使用时有一些要注意的问题，详情见lora_deepspeed/README.md
#### lora_fsdp
    pytorch fsdp 的 zero-3，由于使用的chatglm-6b是float16和float32混合的权重类型，
    暂时fsdp不支持混合类型的模型加载和分片。所以无法运行，只提供代码的参考。
