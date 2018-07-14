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
        - [1.1.2 确定性策略](#112-确定性策略)
        - [1.1.3 对比](#113-对比)
    - [1.2 AC框架](#12-ac框架)
        - [1.2.1 随机策略AC方法](#121-随机策略ac方法)
        - [1.2.2 确定性策略AC方法（DPG）](#122-确定性策略ac方法dpg)
        - [1.2.3 深度确定性策略梯度方法（DDPG）](#123-深度确定性策略梯度方法ddpg)

<!-- /TOC -->


参考**《深入浅出强化学习》**

## 1. 理论基础

model-free的策略搜索方法可以分为随机策略搜索方法和确定性策略搜索方法。

+ 2014年以前，学者们都在发展随机策略搜索方法。因为大家认为确定性策略梯度是不存在的。
+ 2014年Silver在论文[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)中提出了**确定性策略理论**，即DPG。
+ 2015年DeepMind又将DPG与DQN的成功经验相结合，提出了[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)，即**DDPG**
+ ICML2016，提出了[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)，即A3C（asynchronous advantage actor-critic）算法。

### 1.1 随机策略与确定性策略

#### 1.1.1 随机策略

随机策略公式为：

`\[
\pi_{\theta}(a|s)=P[a|s;\theta]
\]`

含义为，在状态`\(s\)`时，动作符合参数为`\(\theta\)`的概率分布，例如常用的高斯策略：

`\[
\pi_{\theta}(a|s)=\frac{1}{\sqrt{2\pi \sigma}}exp(-\frac{(a-f_{\theta}(s))}{2\sigma ^2})
\]`

在状态`\(s\)`处，采取的动作服从均值为`\(f_{\theta}(s)\)`，方差为`\(\sigma ^2\)`的正态分布。所以，即使相同的食碗面反碗底，每次采取的动作也可能不一样。

#### 1.1.2 确定性策略

确定性策略的公式如下：

`\[
a=\mu_{\theta}(s)
\]`

相同的策略（即相同`\(\theta\)`），在状态`\(s\)`时，动作是唯一确定的。

#### 1.1.3 对比

+ 确定性策略的优点在于：**需要采样的数据少**，算法效率高

随机策略的梯度计算公式：

`\[
\triangledown _{\theta}J(\pi _{\theta})=E_{s\sim \rho ^{\pi},a\sim \pi_{\theta}}[\triangledown _{\theta}log\pi_{\theta}(a|s)Q^{\pi}(s,a)]
\]`

其中的`\(Q^{\pi}(s,a)\)`是状态-行为值函数。可见，策略梯度是关于状态和动作的期望，在求期望时，需要对状态分布和动作分布求积分，需要在状态空间和动作空间内大量采样，这样求出来的均值才能近似期望。

而确定性策略的动作是确定的，所以，如果存在确定性策略梯度，其求解**不需要在动作空间采样**，所以需要的样本数更少。对于动作空间很大的智能体（如多关节机器人），动作空间维数很大，有优势。

+ 随机策略的优点：随机策略可以将**探索和改善**集成到**一个策略**中

随机策略本身自带探索，可以通过探索产生各种数据（有好有坏），好的数据可以让强化学习算法改进当前策略。

而确定性策略给定状态和策略参数时，动作是固定的，所以无法探索其他轨迹或者访问其他状态。

**确定性策略**无法探索环境，所以需要通过**异策略**（off-policy）方法来进行学习，即行动策略和评估策略不是同一个策略。**行动策略采用随机策略，而评估策略我要用确定性策略**。而整个确定性策略的学习框架采用的是AC方法。

### 1.2 AC框架

这里会参考[https://blog.csdn.net/jinzhuojun/article/details/72851548](https://blog.csdn.net/jinzhuojun/article/details/72851548)

Actor-Critic（AC）方法其实是policy-based和value-based方法的结合。因为它本身是一种PG方法，同时又结合了value estimation方法，所以有些地方将之归为PG方法的一种，有些地方把它列为policy-based和value-based以外的另一种方法。

+ Actor指的是行动策略，负责policy gradient学习策略
+ Critic指的是评估策略，负责policy evaluation估计value function

所以，

+ 一方面actor学习策略，而策略更新依赖critic估计的value function；
+ 另一方面critic估计value function，而value function又是策略的函数。

如果是Actor-only，那就是policy gradient，而如果是Critic-only，那就是Q-learning。

#### 1.2.1 随机策略AC方法

随机策略的梯度为

`\[
\triangledown _{\theta}J(\pi _{\theta})=E_{s\sim \rho ^{\pi},a\sim \pi_{\theta}}[\triangledown _{\theta}log\pi_{\theta}(a|s)Q^{\pi}(s,a)]
\]`

其中Actor方法用来调整`\(\theta\)`值，

Critic方法逼近值函数`\(Q^{w}(s,a)\approx Q^{\pi}(s,a)\)`，其中`\(w\)`为待逼近的参数，可以用TD学习的方法来评估值函数。

异策略随机梯度为

`\[
\triangledown _{\theta}J(\pi _{\theta})=E_{s\sim \rho ^{\pi},a\sim \beta}[\frac{\pi_{\theta}(a|s)}{\beta_{\theta}(a|s)}\triangledown _{\theta}log\pi_{\theta}(a|s)Q^{\pi}(s,a)]
\]`

和原公式的区别在于采样策略为`\(\beta\)`，即`\(a\sim \beta\)`，从而，多了一项`\(\frac{\pi_{\theta}(a|s)}{\beta_{\theta}(a|s)}\)`。

#### 1.2.2 确定性策略AC方法（DPG）

确定性的策略梯度为：

`\[
\triangledown _{\theta}J(\mu _{\theta})=E_{s\sim \rho ^{\mu}}[\triangledown _{\theta}\mu_{\theta}(s)\triangledown _{\theta}Q^{\mu}(s,a)|_{a=\mu_{\theta}(s)}]
\]`

可见，区别如下：

+ `\(\pi_{\theta}\)`变成了`\(\mu_{\theta}\)`
+ 原来的`\(Q^{\pi}(s,a)\)`改成了`\(Q^{\mu}(s,a)|_{a=\mu_{\theta}(s)}\)`
+ 原来的`\(s\sim \rho ^{\pi}\)`变成了`\(s\sim \rho ^{\mu}\)`
+ 去掉了对于动作的采样`\(a\sim \pi _{\theta}\)`，而改成确定性的动作`\(a=\mu_{\theta}(s)\)`


#### 1.2.3 深度确定性策略梯度方法（DDPG）

