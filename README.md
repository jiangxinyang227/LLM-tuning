### 大模型微调
    提供不同大模型的微调代码，实现在单GPU，多GPU上数据并行，模型并行的微调。

#### 安装环境
* sh install_env.sh
* 要注意torch的版本，cuda的版本，transformers的版本，详情见install_env.sh。因为我的测试环境是在容器里，可能还会有另外包需要安装，运行时缺啥就直接安装，核心就这几个。

#### chatglm_tuning
    chatglm-6b的微调
#### llama_tuning
    llama-7b的微调
#### rlhf_tuning
基于[DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)框架在bloom上的微调。
