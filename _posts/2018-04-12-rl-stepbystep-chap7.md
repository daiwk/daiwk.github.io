---
layout: post
category: "rl"
title: "深入浅出强化学习-chap7 基于策略梯度的强化学习方法"
tags: [深入浅出强化学习, PG, policy gradient, 策略梯度]
---

目录

<!-- TOC -->

- [1. 概述](#1-概述)
    - [似然率的角度](#似然率的角度)
    - [重要性采样的角度](#重要性采样的角度)

<!-- /TOC -->


参考**《深入浅出强化学习》**

## 1. 概述

### 似然率的角度

+ `\(\tau\)`表示一组状态-行为序列（轨迹）`\(s_0,u_0,...,s_H,u_H\)`
+ `\(R(\tau) = \sum_{t=0}^{H}R(s_t,u_t)\)`表示这条轨迹的回报
+ `\(P(\tau;\theta)\)`表示轨迹`\(\tau\)`出现的概率

那么，强化学习的**优化目标**就是长期累积期望回报：

`\[
U(\theta) = E(\sum^{H}_{t=0}R(s_t,u_t);\pi_{\theta})=\sum_{\tau}P(\tau;\theta)R(\tau)
\]`

所以强化学习就是要找到最优参数`\(\theta\)`，使得`\(\max_{\theta}U(\theta)=\max_{\theta}\sum_{\tau}P(\tau;\theta)R(\tau)\)`，那就可以用**梯度上升**（因为是**求max**）来解了。

关键在于如何对`\(U(\theta)\)`求导：

`\[
\begin{split}
\\\\triangledown _{\theta}(U_{\theta})&=1
\\&=2
\\&=3
\\&=4
\end{split}
\]`

### 重要性采样的角度




