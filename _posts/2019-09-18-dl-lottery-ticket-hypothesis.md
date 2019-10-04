---
layout: post
category: "dl"
title: "The Lottery Ticket Hypothesis"
tags: [lottery ticket hypothesis, 彩票, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

参考[训练网络像是买彩票？神经网络剪枝最新进展之彩票假设解读](https://mp.weixin.qq.com/s/Qxeh89AADpZZomizurgpEw)

ICLR2019，[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)

彩票假设的正式定义为：一个随机初始化的密集神经网络包含一个初始化的子网络，在单独训练时，最多经过相同的迭代次数，可以达到和原始网络一样的测试准确率。

我们将一个复杂网络的所有参数当做奖池，上述一组子参数对应的子网络就是中奖彩票。

作者提出了彩票假设并给出一种寻找中奖彩票的方法，通过迭代非结构化剪枝的方式可以找到一个子网络，用原始网络的初始化参数来初始化，可以在性能不下降的情况下更快的训练这个子网络，但是如果用随机初始化方法却达不到同样的性能。

作者也在文章中指出这项工作存在的一些问题。例如，迭代剪枝的计算量太大，需要对一个网络进行连续 15 次或 15 次以上的多次训练。未来可以探索更加高效的寻找中奖彩票的方法。

[Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://arxiv.org/abs/1905.01067)

