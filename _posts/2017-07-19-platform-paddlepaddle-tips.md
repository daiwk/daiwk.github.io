---
layout: post
category: "platform"
title: "paddle相关tips"
tags: [paddle, tips]
---

目录

<!-- TOC -->

- [1. tips](#1-tips)
    - [1.1 学习率](#11-学习率)
    - [1.2 日志解释](#12-日志解释)
    - [1.3 集群设置](#13-集群设置)
    - [1.4 正则](#14-正则)
    - [1.5 初始化](#15-初始化)

<!-- /TOC -->

## 1. tips

### 1.1 学习率

相关issue [https://github.com/PaddlePaddle/Paddle/issues/1167](https://github.com/PaddlePaddle/Paddle/issues/1167)

paddle_pr/proto/TrainerConfig.proto中有关于momentum各个参数的解释

学习率太大可能会出现floating point exception，可以调小一个数量级，或者使用[梯度clipping](https://github.com/PaddlePaddle/models/blob/develop/nmt_without_attention/train.py#L35)

### 1.2 日志解释

Q: 日志中打出来的是 训练误差还是测试误差？ 

A: 训练的时候会打印训练误差，测试的时候会打印测试误差


### 1.3 集群设置

+ 应该要保证文件数目大于节点数
+ 单个节点的trainer数要小于batch size

### 1.4 正则

虽然损失函数里有正则，但在神经网络真正实现时，正则和损失函数是分开的两个部分。并不需要显示的将正则定义在损失函数里面。

全局化的正则（会加给每一个参数），设置方法如下：
https://github.com/PaddlePaddle/models/blob/develop/mt_with_external_memory/train.py#L73

如果想针对性的调整某一个参数的正则，每个可学习的参数都可以通过 param_attr 关键字来设置一些如初始化 mead、std、正则系数等关键参数，调用方式可以参考这个例子：https://github.com/PaddlePaddle/models/blob/develop/text_classification/network_conf.py#L46

如果在optimizer中设置了全局正则系数，下面调用可以取消对fc层参数的正则，当然，也可以设置一个非0值，调整这个参数的正则系数。

```
 fc = paddle.layer.fc(
                 input=input_layer,
                 size=128,
                 param_attr=paddle.attr.Param(decay_rate=0))
```

### 1.5 初始化

正交初始化：

```python
W = np.random.randn(ndim, ndim)
u, s, v = np.linalg.svd(W)
```
求正交要 svd ，可以用numpy算好之后，用u做初始化

