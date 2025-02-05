
# 推理层面

## KVCache对推理的加速方法

## 多头潜在注意力(MLA)如何压缩KVCache的显存占用

# 介绍各种推理引擎常用的加速推理方法

## Persistent Batch（也叫continuous batching）

说简单点就是用一个模型同时推理多个序列，增加模型吞吐量

## KV Cache重用

DeepSeek也使用了这种方法，最直观的感受是命中缓存的部分价格是 百万tokens/0.1元，便宜了一个数量级。
![image](https://github.com/user-attachments/assets/2b819478-b361-4278-8add-54cbd1555121)

KV Cache重用的方法可以参考SGLang的RadixAttention，最核心的思想就是具有相同前缀的输入可以共享KV Cache
SGLang论文：https://arxiv.org/abs/2312.07104
![image](https://github.com/user-attachments/assets/40b6bf73-3ce1-4b4d-8384-06f95c4ce06d)
KV Cache共享示例，蓝色框是可共享的部分，绿色框是不可共享的部分，黄色框是不可共享的模型输出。

# 量化层面

## 简单介绍下什么是量化，以及介绍下基本的量化思路
介绍点经典工作，如GPTQ，AWQ

## KVCache量化

## 1.58bit的BitNet
这个算法简单到让我头皮发麻，要不是真有人跑通了，我都不敢信这样量化真能跑。

# 模型层面

## MoE架构如何减少激活值来加速计算
