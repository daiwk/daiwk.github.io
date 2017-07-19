---
layout: post
category: "ml"
title: "机器学习相关tips"
tags: [ml, tips]
---


## 1. tips

### 1.1 学习率

相关issue [https://github.com/PaddlePaddle/Paddle/issues/1167 ]https://github.com/PaddlePaddle/Paddle/issues/1167 (https://github.com/PaddlePaddle/Paddle/issues/1167)

paddle_pr/proto/TrainerConfig.proto中有关于momentum各个参数的解释

### 1.2 日志解释

Q: 日志中打出来的是 训练误差还是测试误差？ 

A: 训练的时候会打印训练误差，测试的时候会打印测试误差

