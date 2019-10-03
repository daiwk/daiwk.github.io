---
layout: post
category: "platform"
title: "多gpu并行"
tags: [多gpu并行, ]
---

目录

<!-- TOC -->


<!-- /TOC -->


[Optimizing Multi-GPU Parallelization Strategies for Deep Learning Training](https://arxiv.org/pdf/1907.13257.pdf)

参考[分布式训练中数据并行远远不够，「模型并行+数据并行」才是王道](https://mp.weixin.qq.com/s/P67T7uKNzXN3SpfhFv61WQ)

数据并行化（Data parallelism，DP）是应用最为广泛的并行策略，但随着数据并行训练设备数量的增加，设备之间的通信开销也在增长。

此外，每一个训练步中批大小规模的增加，使得模型统计效率（statistical efficiency）出现损失，即获得期望准确率所需的训练 epoch 增加。这些因素会影响整体的训练时间，而且当设备数超出一定量后，利用 DP 获得的加速无法实现很好的扩展。除 DP 以外，训练加速还可以通过模型并行化（model parallelism，MP）实现。

来自加州大学洛杉矶分校和英伟达的研究人员探索了混合并行化方法，即每一个数据并行化 worker 包含多个设备，利用模型并行化分割模型数据流图（model dataflow graph，DFG）并分配至多个设备上。

