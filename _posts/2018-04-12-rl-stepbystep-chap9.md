---
layout: post
category: "rl"
title: "深入浅出强化学习-chap9 基于确定性策略搜索的强化学习方法"
tags: [深入浅出强化学习, DPG, DDPG]
---

目录

<!-- TOC -->

- [1. 理论基础](#1-理论基础)
    - [1.1 随机策略与确定性策略](#11-随机策略与确定性策略)
        - [1.1.1 随机策略](#111-随机策略)

<!-- /TOC -->



参考**《深入浅出强化学习》**

## 1. 理论基础


model-free的策略搜索方法可以分为随机策略搜索方法和确定性策略搜索方法。

+ 2014年以前，学者们都在发展随机策略搜索方法。因为大家认为确定性策略梯度是不存在的。
+ 2014年Silver在论文[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)中提出了**确定性策略理论**。
+ 2015年DeepMind又将DPG与DQN的成功经验相结合，提出了[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)，即**DDPG**


### 1.1 随机策略与确定性策略

#### 1.1.1 随机策略

随机策略公式为：

`\[
\pi_{\theta}(a|s)=P[a|s;\theta]
\]`

含义为，在状态`\(s\)`时，动作符合参数为`\(\theta\)`的概率分布，例如常用的高斯策略：

`\[
\pi_{\theta}(a|s)=\frac{1}{\sqrt{2\pi \sigma}}exp(-\frac{(a-f_{\theta})(s)}{2\sigma ^2})
\]`

在状态`\(s\)`处，采取的动作服从均值为`\(f_{\theta}(s)\)`，方差为`\(\sigma ^2\)`的正态分布。

