---
layout: post
category: "dl"
title: "自然语言处理中的自注意力机制（Self-Attention Mechanism）"
tags: [自注意力, self-attention,  ]
---

目录

<!-- TOC -->


<!-- /TOC -->

attention is all you need的解读可以参考

[https://daiwk.github.io/posts/platform-tensor-to-tensor.html](https://daiwk.github.io/posts/platform-tensor-to-tensor.html)

各种attention model可以参考：

[https://daiwk.github.io/posts/dl-attention-models.html](https://daiwk.github.io/posts/dl-attention-models.html)

本文参考[自然语言处理中的自注意力机制（Self-Attention Mechanism）](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247488035&idx=1&sn=9d0568f58cd85d628fa60ddc33d266e9&chksm=96e9cda3a19e44b5e7ce784d08508ad6d03dcd93c96491dd660af4312b9c67b67457486475ea&mpshare=1&scene=1&srcid=0328RMAtTkf2hZSuXZD5vJBR&pass_ticket=tNNNXIGOajFyoVTQkCkEGcrVM4xaK5lnuItOaXnqkjfkBuTkVoKCva7UoF68PTww#rd)

论文作者之一Lukasz Kaiser的ppt：[https://daiwk.github.io/assets/attention-is-all-you-need-lkaiser.pdf](https://daiwk.github.io/assets/attention-is-all-you-need-lkaiser.pdf)

Attention函数的本质可以被描述为**一个查询（query）与一系列（键key-值value）对一起映射成一个输出**。分为以下3步：

+ 将**query**和**每个key**进行**相似度**计算得到权重，常用的相似度函数有点积，拼接，感知机等
+ 使用一个**softmax**函数对这些**权重进行归一化**
+ 将**权重**和**相应的键值value**进行**加权求和**得到最后的Attention

目前在**NLP研究**中，key和value常常都是同一个，即 **key=value**。








