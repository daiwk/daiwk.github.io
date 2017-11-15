---
layout: post
category: "cv"
title: "LS-GAN"
tags: [lsgan, Loss-sensitive gan]
---

目录

<!-- TOC -->


<!-- /TOC -->

参考：[知乎专栏：条条大路通罗马LS-GAN：把GAN建立在Lipschitz密度上](https://zhuanlan.zhihu.com/p/25204020)

代码：[https://github.com/guojunq/lsgan](https://github.com/guojunq/lsgan)


这个对f-函数的Lipschitz连续假设，就是沟通LS-GAN和WGAN的关键，因为LS-GAN就是为了限制GAN的无限建模能力而提出的。无限建模能力正是一切麻烦的来源。LS-GAN就是希望去掉这个麻烦

LS-GAN可以看成是使用成对的（Pairwise）“真实/生成样本对”上的统计量来学习f-函数。这点迫使真实样本和生成样本必须相互配合，从而更高效的学习LS-GAN。
如果生成的样本和真实样本已经很接近，我们就不必要求他们的L-函数非得有个固定间隔，因为，这个时候生成的样本已经非常好了，接近或者达到了真实样本水平。
这样呢，LS-GAN就可以集中力量提高那些距离真实样本还很远，真实度不那么高的样本上了。这样就可以更合理使用LS-GAN的建模能力。在后面我们一旦限定了建模能力后，也不用担心模型的生成能力有损失了。这个我们称为“按需分配”。

我们证明了，WGAN在对f-函数做出Lipschitz连续的约束后，其实也是将生成样本的密度假设为了Lipschiz 密度。这点上，和LS-GAN是一致的！两者都是建立在Lipschitz密度基础上的生成对抗网络。

好了，这样我们就给出了WGAN分析梯度消失时候，缺失的哪个定量分析了。
