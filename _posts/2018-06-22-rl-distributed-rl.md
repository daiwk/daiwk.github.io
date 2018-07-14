---
layout: post
category: "rl"
title: "分布式强化学习框架"
tags: [分布式强化学习, A3C, ape-x, rudder]
---

目录

<!-- TOC -->

- [1. 概述](#1-%E6%A6%82%E8%BF%B0)
- [2. A3C](#2-a3c)
- [3. rainbow](#3-rainbow)
- [4. APE-X](#4-ape-x)
- [5. rudder](#5-rudder)

<!-- /TOC -->


## 1. 概述

略
大部分在ray中都已经有啦，还有openai的baselines

[https://github.com/ray-project/ray/blob/master/doc/source/rllib-algorithms.rst](https://github.com/ray-project/ray/blob/master/doc/source/rllib-algorithms.rst)

[https://github.com/openai/baselines](https://github.com/openai/baselines)

## 2. A3C

参考[https://daiwk.github.io/posts/rl-stepbystep-chap9.html](https://daiwk.github.io/posts/rl-stepbystep-chap9.html)

## 3. rainbow

[Rainbow: Combining improvements in deep reinforcement learning](https://arxiv.org/abs/1710.02298)

## 4. APE-X

参考[最前沿：当我们以为Rainbow就是Atari游戏的巅峰时，Ape-X出来把Rainbow秒成了渣！](https://zhuanlan.zhihu.com/p/36375292)

[Distributed Prioritized Experience Replay](https://openreview.net/pdf?id=H1Dy---0Z)

## 5. rudder

参考[比TD、MC、MCTS指数级快，性能超越A3C、DDQN等模型，这篇RL算法论文在Reddit上火了](https://www.jiqizhixin.com/articles/2018-06-22-3)

在强化学习中，**延迟奖励**的存在会严重影响性能，主要表现在随着延迟步数的增加，**对时间差分（TD）估计偏差的纠正时间的指数级增长**，和**蒙特卡洛（MC）估计方差的指数级增长**。针对这一问题，来自奥地利约翰开普勒林茨大学 LIT AI Lab 的研究者提出了一种**基于返回值分解**的新方法 RUDDER。实验表明，RUDDER 的速度是 TD、MC 以及 MC 树搜索（MCTS）的指数级，并在特定 Atari 游戏的训练中很快超越 rainbow、A3C、DDQN 等多种著名强化学习模型的性能。

[RUDDER: Return Decomposition for Delayed Rewards](https://arxiv.org/abs/1806.07857)

源码：[https://github.com/ml-jku/baselines-rudder](https://github.com/ml-jku/baselines-rudder)
