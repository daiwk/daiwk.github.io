---
layout: post
category: "knowledge"
title: "合辑"
tags: [合辑, ]
---

目录

<!-- TOC -->

- [基础研究](#%e5%9f%ba%e7%a1%80%e7%a0%94%e7%a9%b6)
  - [优化方法](#%e4%bc%98%e5%8c%96%e6%96%b9%e6%b3%95)
  - [调参](#%e8%b0%83%e5%8f%82)
  - [新的结构](#%e6%96%b0%e7%9a%84%e7%bb%93%e6%9e%84)
    - [NALU](#nalu)
- [CV](#cv)
  - [cnn优化](#cnn%e4%bc%98%e5%8c%96)
  - [sota-cv](#sota-cv)
    - [半弱监督](#%e5%8d%8a%e5%bc%b1%e7%9b%91%e7%9d%a3)
- [NLP](#nlp)
  - [sota-nlp](#sota-nlp)
  - [对话](#%e5%af%b9%e8%af%9d)
    - [convai](#convai)
- [多目标](#%e5%a4%9a%e7%9b%ae%e6%a0%87)
- [推荐](#%e6%8e%a8%e8%8d%90)
- [ctr](#ctr)
  - [传统ctr](#%e4%bc%a0%e7%bb%9fctr)
  - [深度学习ctr](#%e6%b7%b1%e5%ba%a6%e5%ad%a6%e4%b9%a0ctr)
- [GNN](#gnn)
  - [开源图计算平台](#%e5%bc%80%e6%ba%90%e5%9b%be%e8%ae%a1%e7%ae%97%e5%b9%b3%e5%8f%b0)
- [RL](#rl)
- [压缩与部署](#%e5%8e%8b%e7%bc%a9%e4%b8%8e%e9%83%a8%e7%bd%b2)

<!-- /TOC -->

## 基础研究

### 优化方法

[On Empirical Comparisons of Optimizers for Deep Learning](https://arxiv.org/pdf/1910.05446.pdf)

摘要：优化器选择是当前深度学习管道的重要步骤。在本文中，研究者展示了优化器比较对元参数调优协议的灵敏度。研究结果表明，在解释文献中由最近实证比较得出的排名时，元参数搜索空间可能是唯一最重要的因素。但是，当元参数搜索空间改变时，这些结果会相互矛盾。随着调优工作的不断增加，更一般的优化器性能表现不会比近似于它们的那些优化器差，但最近比较优化器的尝试要么假设这些包含关系没有实际相关性，要么通过破坏包含的方式限制元参数。研究者在实验中发现，优化器之间的包含关系实际上很重要，并且通常可以对优化器比较做出预测。具体来说，流行的自适应梯度方法的性能表现绝不会差于动量或梯度下降法。

推荐：如何选择优化器？本文从数学角度论证了不同优化器的特性，可作为模型构建中的参考资料。

### 调参

[你有哪些deep learning（rnn、cnn）调参的经验？](https://www.zhihu.com/question/41631631/)

### 新的结构

#### NALU

[Measuring Arithmetic Extrapolation Performance](https://arxiv.org/abs/1910.01888)

摘要：神经算术逻辑单元（NALU）是一种神经网络层，可以学习精确的算术运算。NALU 的目标是能够进行完美的运算，这需要学习到精确的未知算术问题背后的底层逻辑。评价 NALU 性能是非常困难的，因为一个算术问题可能有许多种类的解法。因此，单实例的 MSE 被用于评价和比较模型之间的表现。然而，MSE 的大小并不能说明是否是一个正确的方法，也不能解释模型对初始化的敏感性。因此，研究者推出了一种「成功标准」，用来评价模型是否收敛。使用这种方法时，可以从很多初始化种子上总结成功率，并计算置信区间。通过使用这种方法总结 4800 个实验，研究者发现持续性的学习算术推导是具有挑战性的，特别是乘法。

推荐：尽管神经算术逻辑单元的出现说明了使用神经网络进行复杂运算推导是可行的，但是至今没有一种合适的评价神经网络是否能够成功收敛的标准。本文填补了这一遗憾，可供对本领域感兴趣的读者参考

## CV

### cnn优化

[卷积神经网络性能优化](https://zhuanlan.zhihu.com/p/80361782)

### sota-cv

#### 半弱监督

[10亿照片训练，Facebook半弱监督训练方法刷新ResNet-50 ImageNet基准测试](https://mp.weixin.qq.com/s/t1Js479ZRDAw1XzPdx_nQA)

[https://github.com/facebookresearch/semi-supervised-ImageNet1K-models](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models)

[https://ai.facebook.com/blog/billion-scale-semi-supervised-learning](https://ai.facebook.com/blog/billion-scale-semi-supervised-learning)

Facebook将该方法称为“半弱监督”(semi-weak supervision)，是结合了半监督学习和弱监督学习者两种不同训练方法的有点的一种新方法。通过使用teacher-student模型训练范式和十亿规模的弱监督数据集，它为创建更准确、更有效的分类模型打开了一扇门。如果弱监督数据集(例如与公开可用的照片相关联的hashtags)不能用于目标分类任务，该方法还可以利用未标记的数据集来生成高度准确的半监督模型。

## NLP

[Stabilizing Transformers for Reinforcement Learning](https://arxiv.org/abs/1910.06764)

摘要：得益于预训练语言模型强大的能力，这些模型近来在 NLP 任务上取得了一系列的成功。这需要归功于使用了 transformer 架构。但是在强化学习领域，transformer 并没有表现出同样的能力。本文说明了为什么标准的 transformer 架构很难在强化学习中优化。研究者同时提出了一种架构，可以很好地提升 transformer 架构和变体的稳定性，并加速学习。研究者将提出的架构命名为 Gated Transformer-XL (GTrXL)，该架构可以超过 LSTM，在多任务学习 DMLab-30 基准上达到 SOTA 的水平。

推荐：本文是 DeepMind 的一篇论文，将强化学习和 Transformer 结合是一种新颖的方法，也许可以催生很多相关的交叉研究。

### sota-nlp

### 对话

#### convai

[GitHub超1.5万星NLP团队热播教程：使用迁移学习构建顶尖会话AI](https://mp.weixin.qq.com/s/lHzZjY98WxNeQDjTH7VXAw)

[https://convai.huggingface.co/](https://convai.huggingface.co/)

[https://github.com/huggingface/transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai)

====here

## 多目标

[阿里提出多目标优化全新算法框架，同时提升电商GMV和CTR](https://mp.weixin.qq.com/s/JXW--wzpaFwRHSSvZEA0mg)

## 推荐



## ctr

### 传统ctr

[https://daiwk.github.io/posts/dl-traditional-ctr-models.html](https://daiwk.github.io/posts/dl-traditional-ctr-models.html)

### 深度学习ctr

[https://daiwk.github.io/posts/dl-dl-ctr-models.html](https://daiwk.github.io/posts/dl-dl-ctr-models.html)

## GNN

### 开源图计算平台

[https://daiwk.github.io/posts/platform-gnn-frameworks.html](https://daiwk.github.io/posts/platform-gnn-frameworks.html)


## RL

## 压缩与部署

[GDP：Generalized Device Placement for Dataflow Graphs](https://arxiv.org/pdf/1910.01578.pdf)

大型神经网络的运行时间和可扩展性会受到部署设备的影响。随着神经网络架构和异构设备的复杂性增加，对于专家来说，寻找合适的部署设备尤其具有挑战性。现有的大部分自动设备部署方法是不可行的，因为部署需要很大的计算量，而且无法泛化到以前的图上。为了解决这些问题，研究者提出了一种高效的端到端方法。该方法基于一种可扩展的、在图神经网络上的序列注意力机制，并且可以迁移到新的图上。在不同的表征深度学习模型上，包括 Inception-v3、AmoebaNet、Transformer-XL 和 WaveNet，这种方法相比人工方法能够取得 16% 的提升，以及比之前的最好方法有 9.2% 的提升，在收敛速度上快了 15 倍。为了进一步减少计算消耗，研究者在一系列数据流图上预训练了一个策略网络，并使用 superposition 网络在每个单独的图上微调，在超过 50k 个节点的大型图上得到了 SOTA 性能表现，例如一个 8 层的 GNMT。

推荐：本文是谷歌大脑的一篇论文，通过图网络的方法帮助将模型部署在合适的设备上。推荐收到硬件设备限制，需要找到合适部署图的方法的读者参考。

