---
layout: post
category: "rl"
title: "深入浅出强化学习-chap6 基于值函数逼近的强化学习方法"
tags: [深入浅出强化学习, DQN]
---

目录

<!-- TOC -->

- [1. 基于值函数逼近的理论讲解](#1-基于值函数逼近的理论讲解)
- [2. DQN及其变种](#2-dqn及其变种)
    - [2.1 DQN方法](#21-dqn方法)
    - [2.2 Double DQN](#22-double-dqn)
    - [2.3 优先回放(Prioritized Replay)](#23-优先回放prioritized-replay)
    - [2.4 Dueling DQN](#24-dueling-dqn)
- [3. 函数逼近方法](#3-函数逼近方法)
    - [3.1 基于非参数的函数逼近](#31-基于非参数的函数逼近)
        - [基于核函数的方法](#基于核函数的方法)
        - [基于高斯过程的函数逼近方法](#基于高斯过程的函数逼近方法)
    - [3.2 基于参数的函数逼近](#32-基于参数的函数逼近)
    - [3.3 卷积神经网络](#33-卷积神经网络)
        - [3.3.1 卷积运算](#331-卷积运算)
            - [稀疏连接](#稀疏连接)
            - [权值共享](#权值共享)
        - [3.3.2 池化](#332-池化)

<!-- /TOC -->



参考**《深入浅出强化学习》**

## 1. 基于值函数逼近的理论讲解

## 2. DQN及其变种

### 2.1 DQN方法

DeepMind发表在Nature上的文章[Human-level control through deep reinforcement learning](https://daiwk.github.io/assets/dqn.pdf)

最主要创新点是两个：

+ 经验回放
+ 设立单独的目标网络

大体框架是复用**传统强化学习**里的**Qlearning**方法。Qlearning包括两个关键点：

+ 异策略：**行动策略**与 **要评估的策略**不是同一个策略。
    + 行动策略（用来选择行动`\(a\)`的策略）是`\(\epsilon -greedy\)`策略
    + 要评估和改进的策略是贪婪策略（即`\(max_aQ(s_{t+1},a)\)`，当前状态`\(s_{t+1}\)`下，使用各种a使`\(Q(s_{t+1},a)\)`达到的最大值）
+ 时间差分（Time Differential, TD）：使用时间差分目标（即，`\(r_t+\gamma max_aQ(s_{t+1},a)\)`）来更新当前的行为值函数

>初始化`\(Q(s,a),\forall s\in S,a\in A(s)\)`，给定参数`\(\alpha, \gamma\)`
>Repeat
>给定起始状态`\(s\)`，并根据`\(\epsilon\)`-greedy策略在状态`\(s\)`选择动作`\(a\)`
>     Repeat
>         1. 根据`\(\epsilon \)`-greedy策略选择动作`\(a_t\)`，得到回报`\(r_t\)`和下一个状态`\(s_{t+1}\)`
>         1. 使用时间差分方法更新行为值函数`\(Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha [r_t+\gamma max_a(Q(s_{t+1},a))-Q(s_t,a_t)]\)`
>         1. `\(s=s'\)`, `\(a=a'\)`
>     Until s是最终状态
>Until 所有的`\(Q(s,a)\)`收敛
>输出最终策略：`\(\pi (s)=argmax_aQ(s,a)\)`

### 2.2 Double DQN

### 2.3 优先回放(Prioritized Replay)

### 2.4 Dueling DQN

## 3. 函数逼近方法

### 3.1 基于非参数的函数逼近

#### 基于核函数的方法

#### 基于高斯过程的函数逼近方法

### 3.2 基于参数的函数逼近

### 3.3 卷积神经网络

#### 3.3.1 卷积运算

##### 稀疏连接

##### 权值共享

#### 3.3.2 池化
