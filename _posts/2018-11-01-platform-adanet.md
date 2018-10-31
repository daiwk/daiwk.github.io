---
layout: post
category: "platform"
title: "adanet"
tags: [adanet, ]
---

目录

<!-- TOC -->

- [快速易用](#快速易用)
- [学习保证](#学习保证)
- [可扩展](#可扩展)

<!-- /TOC -->

参考[资源 \| 谷歌开源AdaNet：基于TensorFlow的AutoML框架](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650751000&idx=2&sn=62814e8e19f53655bc75c83f84b0b9e1&chksm=871a8466b06d0d701ce5a35c4ee13eeeab5f9a155bd28474aa745b3938099df7e35d43ee754d&mpshare=1&scene=1&srcid=1101yG3KXEYIqzbTknSmIRQu&pass_ticket=zX4VgptyVP3Mvtqcew6Og6Jwwl4pq45hepOQUq59s21SZj%2BEuwQXeeT1wVMIDy0b#rd)


相关论文： [AdaNet: Adaptive Structural Learning of Artificial Neural Networks](http://proceedings.mlr.press/v70/cortes17a/cortes17a.pdf)

Github 项目地址：[https://github.com/tensorflow/adanet](https://github.com/tensorflow/adanet)

教程 notebook：[https://github.com/tensorflow/adanet/tree/v0.1.0/adanet/examples/tutorials](https://github.com/tensorflow/adanet/tree/v0.1.0/adanet/examples/tutorials)

基于 TensorFlow 的轻量级框架 AdaNet，该框架可以使用少量专家干预来自动学习高质量模型。AdaNet 在谷歌近期的强化学习和基于进化的 AutoML 的基础上构建，快速灵活同时能够提供学习保证（learning guarantee）。重要的是，AdaNet 提供通用框架，不仅能用于学习神经网络架构，还能学习集成架构以获取更好的模型。

AdaNet 易于使用，能够创建高质量模型，节省 ML 从业者在选择最优神经网络架构上所花费的时间，实现学习神经架构作为集成子网络的自适应算法。AdaNet 能够添加不同深度、宽度的子网络，从而创建不同的集成，并在性能改进和参数数量之间进行权衡。

### 快速易用

AdaNet 实现了 TensorFlow Estimator 接口，通过压缩训练、评估、预测和导出极大地简化了机器学习编程。它整合如 TensorFlow Hub modules、TensorFlow Model Analysis、Google Cloud』s Hyperparameter Tuner 这样的开源工具。它支持分布式训练，极大减少了训练时间，使用可用 CPU 和加速器（例如 GPU）实现线性扩展。

AdaNet 将 TensorBoard 无缝集成，以监控子网络的训练、集成组合和性能。AdaNet 完成训练后将导出一个 SavedModel，可使用 TensorFlow Serving 进行部署。

### 学习保证

构建神经网络集成存在多个挑战：最佳子网络架构是什么？重复使用同样的架构好还是鼓励差异化好？虽然具备更多参数的复杂子网络在训练集上表现更好，但也因其极大的复杂性它们难以泛化到未见过的数据上。这些挑战源自对模型性能的评估。我们可以在训练集分留出的数据集上评估模型表现，但是这么做会降低训练神经网络的样本数量。

不同的是，AdaNet 的方法是优化一个目标函数，在神经网络集成在训练集上的表现与泛化能力之间进行权衡。直观上，即仅在候选子网络改进网络集成训练损失的程度超过其对泛化能力的影响时，选择该候选子网络。这保证了：

+ 集成网络的泛化误差受训练误差和复杂度的约束。

+ 通过优化这一目标函数，能够直接最小化这一约束。

优化这一目标函数的实际收益是它能减少选择哪个候选子网络加入集成时对留出数据集的需求。另一个益处是允许使用更多训练数据来训练子网络。

### 可扩展

谷歌认为，创建有用的 AutoML 框架的关键是：研究和产品使用方面不仅能够提供合理的默认设置，还要让用户尝试自己的子网络/模型定义。这样，机器学习研究者、从业者、喜爱者都能够使用 tf.layers 这样的 API 定义自己的 AdaNet adanet.subnetwork.Builder。

已在自己系统中融合 TensorFlow 模型的用户可以轻松将 TensorFlow 代码转换到 AdaNet 子网络中，并使用 adanet.Estimator 来提升模型表现同时获取学习保证。AdaNet 将探索他们定义的候选子网络搜索空间，并学习集成这些子网络。例如，采用 NASNet-A CIFAR 架构的开源实现，把它迁移到一个子网络，经过 8 次 AdaNet 迭代后提高其在 CIFAR-10 上的当前最优结果。

通过固定或自定义 tf.contrib.estimator.Heads，用户可以使用自己定义的损失函数作为 AdaNet 目标函数的一部分来训练回归、分类和多任务学习问题。

用户也可以通过拓展 adanet.subnetwork.Generator 类别，完全定义要探索的候选子网络搜索空间。这使得用户能够基于硬件扩大或缩小搜索空间范围。子网络的搜索空间可以简单到复制具备不同随机种子的同一子网络配置，从而训练数十种具备不同超参数组合的子网络，并让 AdaNet 选择其中一个进入最终的集成模型。

