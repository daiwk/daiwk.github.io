---
layout: post
category: "rl"
title: "深入浅出强化学习-chap7 基于策略梯度的强化学习方法"
tags: [深入浅出强化学习, PG, policy gradient, 策略梯度]
---

目录

<!-- TOC -->

- [问题定义](#问题定义)
    - [似然率的角度](#似然率的角度)
    - [重要性采样的角度](#重要性采样的角度)
- [如何求解log梯度](#如何求解log梯度)
    - [REINFORCE算法](#reinforce算法)

<!-- /TOC -->


参考**《深入浅出强化学习》**

## 问题定义

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
\\ \triangledown _{\theta}(U_{\theta})&=\triangledown_{\theta}\sum_{\tau}P(\tau;\theta)R(\tau)
\\&=\sum_{\tau}\triangledown_{\theta}P(\tau;\theta)R(\tau)
\\&=\sum_{\tau} \frac{P(\tau;\theta)}{P(\tau;\theta)} \triangledown_{\theta}P(\tau;\theta)R(\tau)
\\&=\sum_{\tau} P(\tau;\theta)\frac{\triangledown_{\theta}P(\tau;\theta)R(\tau)}{P(\tau;\theta)}
\\&=\sum_{\tau} P(\tau;\theta)\triangledown_{\theta}\log P(\tau;\theta)R(\tau)
\end{split}
\]`

其中，`\(\triangledown_x\log P(x)=\frac{1}{P(x)}\triangledown_xP(x)\)`

因此，策略梯度最终变成求`\(\triangledown_{\theta}\log P(\tau;\theta)R(\tau)\)`的期望。而这期望，可以通过利用当前策略`\(\pi_{\theta}\)`采样m条轨迹`\(\tau_1,...,\tau_m\)`之后，求平均来近似：

`\[
\triangledown _{\theta}(U_{\theta})\approx \frac{1}{m}\sum_{i=1}^{m}\triangledown_{\theta}\log P(\tau_i;\theta)R(\tau_i)
\]`

也从另一个角度来证明，因为`\(R(\tau)\)`和`\(\theta\)`无关，所以其实也可以这么看：

`\[
\triangledown_{\theta}\log P(\tau;\theta)=\frac{1}{P(\tau;\theta)}\triangledown_{\theta}P(\tau;\theta)
\]`

所以，

`\[
\triangledown_{\theta}P(\tau;\theta)=P(\tau;\theta)\triangledown_{\theta}\log P(\tau;\theta)
\]`

所以

`\[
\begin{split}
\\ \triangledown _{\theta}(U_{\theta})&=\triangledown_{\theta}\sum_{\tau}P(\tau;\theta)R(\tau)
\\&=\sum_{\tau}[\triangledown_{\theta}P(\tau;\theta)]R(\tau)
\\&=\sum_{\tau} [P(\tau;\theta)\triangledown_{\theta}\log P(\tau;\theta)]R(\tau)
\end{split}
\]`  

### 重要性采样的角度

暂时略

## 如何求解log梯度

上面推导出了策略梯度的公式是：

`\[
\triangledown _{\theta}(U_{\theta})\approx \frac{1}{m}\sum_{i=1}^{m}\triangledown_{\theta}\log P(\tau_i;\theta)R(\tau_i)
\]`

如何求解`\(\triangledown_{\theta}\log P(\tau;\theta)\)`呢？

### REINFORCE算法

首先，轨迹`\(\tau=s_0,u_0,...,s_H,u_H\)`，那么：

`\[
P(\tau;\theta)=P(s_0)\prod ^{H-1}_{i=0}P(s_{i+1}|s_{i},u_{i})\pi_{\theta}(u_i|s_i)
\]`

其中，`\(\pi_{\theta}(u_i|s_i)\)`是策略，也就是在状态`\(s_i\)`下，采用动作`\(u_i\)`的概率。

