---
layout: post
category: "dl"
title: "激活函数"
tags: [激活函数, activation function ]
---

目录

<!-- TOC -->

- [1. Step](#1-step)
- [2. Identity](#2-identity)
- [3. ReLU](#3-relu)
- [4. Sigmoid](#4-sigmoid)
- [5. Tanh](#5-tanh)
- [6. Leaky Relu](#6-leaky-relu)
- [7. PReLU](#7-prelu)
- [8. RReLU](#8-rrelu)
- [9. ELU](#9-elu)
- [10. SELU](#10-selu)
    - [概述](#概述)
- [11. SReLU](#11-srelu)
- [12. Hard Sigmoid](#12-hard-sigmoid)
- [13. Hard Tanh](#13-hard-tanh)
- [14. LeCun Tanh](#14-lecun-tanh)
- [15. ArcTan](#15-arctan)
- [16. SoftSign](#16-softsign)
- [17. SoftPlus](#17-softplus)
- [18. Signum](#18-signum)
- [19. Bent Identity](#19-bent-identity)
- [20. Symmetrical Sigmoid](#20-symmetrical-sigmoid)
- [21. Log Log](#21-log-log)
- [22. Gaussian](#22-gaussian)
- [23. Absolute](#23-absolute)
- [24. Sinusoid](#24-sinusoid)
- [25. Cos](#25-cos)
- [26. Sinc](#26-sinc)

<!-- /TOC -->

参考[26种神经网络激活函数可视化](https://www.jiqizhixin.com/articles/2017-10-10-3)，原文[Visualising Activation Functions in Neural Networks](https://dashee87.github.io/data%20science/deep%20learning/visualising-activation-functions-in-neural-networks/)【可以在里面选择不同的激活函数，看图】

## 1. Step

## 2. Identity 

## 3. ReLU

## 4. Sigmoid

## 5. Tanh

## 6. Leaky Relu

## 7. PReLU

## 8. RReLU

## 9. ELU

## 10. SELU

参考
[引爆机器学习圈：「自归一化神经网络」提出新型激活函数SELU](https://zhuanlan.zhihu.com/p/27362891)
[加速网络收敛——BN、LN、WN与selu](http://skyhigh233.com/blog/2017/07/21/norm/)
[如何评价 Self-Normalizing Neural Networks 这篇论文?](https://www.zhihu.com/question/60910412)

paper: [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

### 概述

其实就是ELU乘了个lambda，关键在于这个lambda是大于1的。以前relu，prelu，elu这些激活函数，都是在负半轴坡度平缓，这样在activation的方差过大的时候可以让它减小，防止了梯度爆炸，但是正半轴坡度简单的设成了1。而selu的正半轴大于1，在方差过小的的时候可以让它增大，同时防止了梯度消失。这样激活函数就有一个不动点，网络深了以后每一层的输出都是均值为0方差为1。

## 11. SReLU


## 12. Hard Sigmoid

## 13. Hard Tanh

## 14. LeCun Tanh

## 15. ArcTan

## 16. SoftSign

Softsign 是 Tanh 激活函数的另一个替代选择。就像 Tanh 一样，Softsign 是反对称、去中心、可微分，并返回-1 和 1 之间的值。其更平坦的曲线与更慢的下降导数表明它可以更高效地学习。

`\[
f(x)=\frac{x}{1+|x|}
\]`

`\[
f'(x)=\frac{1}{1+|x|^2}
\]`

## 17. SoftPlus

## 18. Signum

## 19. Bent Identity

## 20. Symmetrical Sigmoid

## 21. Log Log

## 22. Gaussian

## 23. Absolute

## 24. Sinusoid

## 25. Cos

## 26. Sinc

