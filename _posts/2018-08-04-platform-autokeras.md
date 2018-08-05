---
layout: post
category: "platform"
title: "autokeras"
tags: [autokeras, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

参考[https://mp.weixin.qq.com/s/NAjaco_dC1y3Eo_-UfAsbg](https://mp.weixin.qq.com/s/NAjaco_dC1y3Eo_-UfAsbg)


官网：[http://autokeras.com/](http://autokeras.com/)

代码：[https://github.com/jhfjhfj1/autokeras](https://github.com/jhfjhfj1/autokeras)

另一篇enas的论文
[Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268)

enas对应的tf和pytorch实现：

+ tf：[https://github.com/melodyguan/enas](https://github.com/melodyguan/enas)
+ pytorch：[https://github.com/carpedm20/ENAS-pytorch](ttps://github.com/carpedm20/ENAS-pytorch)


NAS：

[Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)

神经架构搜索（NAS）是自动机器学习中一种有效的计算工具，旨在为给定的学习任务搜索最佳的神经网络架构。然而，现有的 NAS 算法通常计算成本很高。另一方面，网络态射（network morphism）已经成功地应用于神经架构搜索。网络态射是一种改变神经网络架构但保留其功能的技术。因此，我们可以利用网络态射操作将训练好的神经网络改成新的体系架构，如，插入一层或添加一个残差连接。然后，只需再加几个 epoch 就可以进一步训练新架构以获得更好的性能。

基于网络态射的 NAS 方法要解决的最重要问题是运算的选择，即从网络态射运算集里进行选择，将现有的架构改变为一种新的架构。基于最新网络态射的方法使用深度强化学习控制器，这需要大量的训练样例。另一个简单的方法是使用随机算法和爬山法，这种方法每次只能探索搜索区域的邻域，并且有可能陷入局部最优值。

贝叶斯优化已被广泛用于基于观察有限数据的寻找函数最优值过程。它经常被用于寻找黑箱函数的最优点，其中函数的观察值很难获取。贝叶斯优化的独特性质启发了研究者探索它在指导网络态射减少已训练神经网络数量的能力，从而使搜索更加高效。

为基于网络态射的神经架构搜索设计贝叶斯优化方法是很困难的，因为存在如下挑战：

+ 首先，其潜在的高斯过程（GP）在传统上是用于欧氏空间的，为了用观察数据更新贝叶斯优化，潜在高斯过程将使用搜索到的架构和它们的性能来训练。然而，神经网络架构并不位于欧氏空间，并且很难参数化为固定长度的向量。
+ 其次，采集函数需要进行优化以生成下一个架构用于贝叶斯优化。然而，这个过程不是最大化欧氏空间里的一个函数来态射神经架构，而是选择一个节点在一个树架构搜索空间中扩展，其中每个节点表示一个架构，且每条边表示一个态射运算。传统的类牛顿或基于梯度的方法不能简单地进行应用。第三，网络态射运算改变神经架构的的一个层可能会导致其它层的很多变化，以保持输入和输出的一致性，这在以前的研究中并没有定义。网络态射运算在结合了跳过连接的神经架构搜索空间中是很复杂的。

在 AutoKeras 作者提交的论文中，研究人员们提出了一种带有网络态射的高效神经架构搜索，它利用贝叶斯优化通过每次选择最佳运算来引导搜索空间。为应对上述挑战，研究者创建了一种基于编辑距离（edit-distance）的神经网络核函数。与网络态射的关键思路一致，它给出了将一个神经网络转化为另一个神经网络需要多少运算。此外，研究者为树形架构搜索空间专门设计了一种新的采集函数（acquisition function）优化器，使贝叶斯优化能够从运算中进行选择。优化方法可以在优化过程中平衡探索和利用。此外，作者还定义了一个网络级态射，以解决基于前一层网络态射的神经架构中的复杂变化。该方法被封装成一个开源软件，即 AutoKeras，在基准数据集上进行评估，并与最先进的基线方法进行比较。

论文：[Efficient Neural Architecture Search with Network Morphism](https://arxiv.org/abs/1806.10282)

之前还有一篇enas的文章：


安装：

```shell
pip install autokeras
```

如果装完后提示libgcc_s.so里的GCC_VERSION找不到啥的，可以试一下用root：

```shell
cp /opt/compiler/gcc-4.8.2/lib/libgcc_s.so* /lib64/
```
