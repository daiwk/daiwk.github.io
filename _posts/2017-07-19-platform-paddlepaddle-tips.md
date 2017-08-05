---
layout: post
category: "platform"
title: "paddle相关tips"
tags: [paddle, tips]
---


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