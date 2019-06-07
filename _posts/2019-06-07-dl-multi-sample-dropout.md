---
layout: post
category: "dl"
title: "multi-sample dropout"
tags: [multi-sample dropout, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

[Multi-Sample Dropout for Accelerated Training and Better Generalization](https://arxiv.org/pdf/1905.09788.pdf)

[大幅减少训练迭代次数，提高泛化能力：IBM提出「新版Dropout」](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650763415&idx=5&sn=9c7ccac83e5da883ffb05ca5b6d20954&chksm=871ab4e9b06d3dff984df2ffa38520a6ca6cee546919f527045467c12a26053c2a00fedadf1c&mpshare=1&scene=1&srcid=&pass_ticket=TloMdmvUbLd5jnKvVTzrccQhGuskwL6KQ0HhJLF56Nwtcb16%2BVvMA09bw32tFrjs#rd)

简单地说，假设dropout的比例是0.5，那么dropout会在每轮训练中随机忽略（即 drop）**50%的神经元**，以避免过拟合的发生。如此一来，神经元之间无法相互依赖，从而保证了神经网络的泛化能力。在infer时，会**用到所有的神经元**，因此所有的信息都被保留；但**输出值会乘0.5**，使**平均值与训练时间一致**。这种推理网络可以看作是训练过程中随机生成的**多个子网络的集合**。后来有一些变形，例如DropConnect，也就是[Regularization of Neural Networks using DropConnect](http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf)，随机忽略的是神经元之间的部分连接，而不是神经元。

本文阐述的也是一种 dropout 技术的变形——multi-sample dropout。传统 dropout 在每轮训练时会从输入中随机选择一组样本（称之为 dropout 样本），而 multi-sample dropout 会创建多个 dropout 样本，然后平均所有样本的损失，从而得到最终的损失。这种方法只要在 dropout 层后复制部分训练网络，并在这些复制的全连接层之间共享权重就可以了，无需新运算符。

通过综合 M 个 dropout 样本的损失来更新网络参数，使得最终损失比任何一个 dropout 样本的损失都低。这样做的效果类似于对一个 minibatch 中的每个输入重复训练 M 次。因此，它大大减少了训练迭代次数。