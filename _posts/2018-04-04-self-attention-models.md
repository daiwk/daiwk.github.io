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


Attention 函数的本质可以被描述为一个查询（query）到一系列（键key-值value）对的映射，如下图：

作者之一Lukasz Kaiser的ppt：[https://daiwk.github.io/assets/attention-is-all-you-need-lkaiser.pdf](https://daiwk.github.io/assets/attention-is-all-you-need-lkaiser.pdf)

