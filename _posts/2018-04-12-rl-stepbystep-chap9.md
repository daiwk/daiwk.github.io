---
layout: post
category: "rl"
title: "深入浅出强化学习-chap9 基于确定性策略搜索的强化学习方法"
tags: [深入浅出强化学习, DPG, DDPG]
---

目录

<!-- TOC -->

- [1. 理论基础](#1-%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80)

<!-- /TOC -->



参考**《深入浅出强化学习》**

## 1. 理论基础


model-free的策略搜索方法可以分为随机策略搜索方法和确定性策略搜索方法。

+ 2014年以前，学者们都在发展随机策略搜索方法。因为大家认为确定性策略梯度是不存在的。
+ 2014年Silver在论文[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)中提出了**确定性策略理论**。
+ 2015年DeepMind又将DPG与DQN的成功经验相结合，提出了[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)，即**DDPG**






