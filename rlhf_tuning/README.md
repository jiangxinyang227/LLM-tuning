### 基于DeppSpeed-Chat在Bloom-1B上的微调

#### 微调的注意事项
* 1、GPU内存不足时，可以开启zero stage3 和 offload，但这需要你的CPU内存足够，是一种以CPU补GPU的方法。
* 2、训练奖励模型时可以选择尺寸小的模型，在这里选择了Bloom-560m。
* 3、DeepSpeed-Chat提供的奖励模型的代码只适合于padding_side=right这种情况，如果模型原来的padding_side=left,则在初始化tokenizer之后，需要添加一行tokenizer.padding_side = "right"。
* 4、先微调step1和step2，微调step3时actor和critic两个模型使用step1和step2产出的模型。