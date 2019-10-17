---
layout: post
category: "rl"
title: "强化学习进展（持续更新）"
tags: [强化学习, 进展, 迁移学习,]
---

目录

<!-- TOC -->

- [Beyond DQN/A3C: A Survey in Advanced Reinforcement Learning(2018)](#beyond-dqna3c-a-survey-in-advanced-reinforcement-learning2018)
- [Modern Deep Reinforcement Learning Algorithms(2019)](#modern-deep-reinforcement-learning-algorithms2019)
- [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](#evolution-strategies-as-a-scalable-alternative-to-reinforcement-learning)
- [模拟器相关](#%e6%a8%a1%e6%8b%9f%e5%99%a8%e7%9b%b8%e5%85%b3)
- [代码库](#%e4%bb%a3%e7%a0%81%e5%ba%93)
- [迁移学习+强化学习](#%e8%bf%81%e7%a7%bb%e5%ad%a6%e4%b9%a0%e5%bc%ba%e5%8c%96%e5%ad%a6%e4%b9%a0)

<!-- /TOC -->

梳理rl的一些新进展

## Beyond DQN/A3C: A Survey in Advanced Reinforcement Learning(2018)

参考[深度 \| 超越DQN和A3C：深度强化学习领域近期新进展概览](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650751210&idx=4&sn=a1d4c4c2b27a7f62a4edbd74f626f23c&chksm=871a8494b06d0d82e3dc22b8a591cde7449d845576bbbdecf98ad513a4326cfe9b34eb6272b4&mpshare=1&scene=1&srcid=050208JkhftrjXIfdxVcWRZi&pass_ticket=csFmp%2BqPqpbOEtBCr9byDm0vHyp83ccxf21EyZaHyV%2BoFQOLINXIlgzuTkVvCg24#rd)

原blog：[https://towardsdatascience.com/advanced-reinforcement-learning-6d769f529eb3](https://towardsdatascience.com/advanced-reinforcement-learning-6d769f529eb3)

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


## Evolution Strategies as a Scalable Alternative to Reinforcement Learning

[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)

blog：[https://openai.com/blog/evolution-strategies/](https://openai.com/blog/evolution-strategies/)

代码：[https://github.com/openai/evolution-strategies-starter](https://github.com/openai/evolution-strategies-starter)


## 模拟器相关

[Virtual-Taobao: Virtualizing Real-world Online Retail Environment for Reinforcement Learning](https://arxiv.org/pdf/1805.10000.pdf)

[Simulating User Feedback for Reinforcement Learning Based Recommendations](https://arxiv.org/pdf/1906.11462.pdf)

## 代码库

[https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)

参考[17种深度强化学习算法用Pytorch实现](https://mp.weixin.qq.com/s/BtNH2_s4l1Kcwe82y-oEkA)

## 迁移学习+强化学习

[八千字长文深度解读，迁移学习在强化学习中的应用及最新进展](https://mp.weixin.qq.com/s/Rj55EoopzlR71DZ5XrvH_w)
