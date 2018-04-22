---
layout: post
category: "platform"
title: "tensorflow probability"
tags: [tensorflow probability, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

参考：

参考 [资源 \| 概率编程工具：TensorFlow Probability官方简介](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650741211&idx=2&sn=760f7041d77ad19d9da27c16fd1eff40&chksm=871adda5b06d54b3982d54a17eab28a599f6921e3fff2b49f3465fcb2eca788a46947ee50d26&scene=0&pass_ticket=EUwwyIFBVe43KU36qUZYEq7h15BWgwS7KrwLya%2BB%2B1HtzW6kqMdyvHJs1asAuyqo#rd)

原文：[https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245](https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245)

TensorFlow Probability 适用于以下需求：

+ 希望建立一个**生成模型**，推理其hidden processes。
+ 需要**量化**预测结果的**不确定性（uncertainty）**，而不是预测单个值。
+ 训练集具有大量**与数据点数量相关的特征**。
+ 训练数据是**结构化的**（例如，使用分组，空间，图表或语义），并且想**使用先验信息（prior information）**来捕捉其中的结构。
+ 存在一个inverse problem - 参考 [TFDS'18 演讲视频](https://www.youtube.com/watch?v=Bb1_zlrjo1c)以重建测量中的融合等离子体。



安装：

```shell
pip install --user --upgrade tfp-nightly
```

源码：[https://github.com/tensorflow/probability](https://github.com/tensorflow/probability)

