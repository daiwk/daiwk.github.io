---
layout: post
category: "rl"
title: "强化学习进展（持续更新）"
tags: [强化学习, 进展, ]
---

目录

<!-- TOC -->

- [xxx(2018)](#xxx2018)
    - [Review: DQN and A3C/A2C](#review-dqn-and-a3ca2c)
- [Modern Deep Reinforcement Learning Algorithms(2019)](#modern-deep-reinforcement-learning-algorithms2019)

<!-- /TOC -->

参考[深度 \| 超越DQN和A3C：深度强化学习领域近期新进展概览](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650751210&idx=4&sn=a1d4c4c2b27a7f62a4edbd74f626f23c&chksm=871a8494b06d0d82e3dc22b8a591cde7449d845576bbbdecf98ad513a4326cfe9b34eb6272b4&mpshare=1&scene=1&srcid=050208JkhftrjXIfdxVcWRZi&pass_ticket=csFmp%2BqPqpbOEtBCr9byDm0vHyp83ccxf21EyZaHyV%2BoFQOLINXIlgzuTkVvCg24#rd)

原blog：[https://towardsdatascience.com/advanced-reinforcement-learning-6d769f529eb3](https://towardsdatascience.com/advanced-reinforcement-learning-6d769f529eb3)

## xxx(2018)

### Review: DQN and A3C/A2C

DQN

`\[
Q\left(s_{t}, a_{t} ; \theta\right) \leftarrow Q\left(s_{t}, a_{t} ; \theta\right)+\alpha[\underbrace{\underbrace{(r_{t}+\max _{a} \hat{Q}\left(s_{t+1}, a ; \theta^{\prime}\right))}_{\text { target }}-Q\left(s_{t}, a_{t} ; \theta\right) )}_{\text {TD-error}}]
\]`

ac

`\[
d \theta_{v} \leftarrow d \theta_{v}+\partial{\underbrace{\left(R-V\left(s_{i} ; \theta_{v}\right)\right)}_{\text{advantage}}}^{2} / \partial \theta_{v}
\]`

## Modern Deep Reinforcement Learning Algorithms(2019)

[Modern Deep Reinforcement Learning Algorithms](https://arxiv.org/pdf/1906.10025v1.pdf)

原论文有点大。。打开太慢。。转存一份：[https://daiwk.github.io/assets/Modern%20Deep%20Reinforcement%20Learning%20Algorithms.pdf](https://daiwk.github.io/assets/Modern%20Deep%20Reinforcement%20Learning%20Algorithms.pdf)

