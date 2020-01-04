---
layout: post
category: "knowledge"
title: "合辑"
tags: [合辑, ]
---

目录

<!-- TOC -->

- [基础研究](#基础研究)
    - [优化方法](#优化方法)
    - [调参](#调参)
    - [新的结构](#新的结构)
        - [NALU](#nalu)
- [CV](#cv)
    - [cnn优化](#cnn优化)
    - [sota-cv](#sota-cv)
        - [半弱监督](#半弱监督)
- [NLP](#nlp)
    - [标准](#标准)
    - [bert变种](#bert变种)
        - [gated transformer-xl](#gated-transformer-xl)
        - [bert蒸馏、量化、剪枝](#bert蒸馏量化剪枝)
    - [对话](#对话)
        - [convai](#convai)
- [语音](#语音)
    - [中文语音识别](#中文语音识别)
- [视频](#视频)
    - [快手相关工作](#快手相关工作)
        - [EIUM：讲究根源的快手短视频推荐](#eium讲究根源的快手短视频推荐)
        - [Comyco：基于质量感知的码率自适应策略](#comyco基于质量感知的码率自适应策略)
        - [Livesmart：智能CDN调度](#livesmart智能cdn调度)
    - [google](#google)
- [多目标](#多目标)
- [推荐](#推荐)
- [ctr](#ctr)
    - [传统ctr](#传统ctr)
    - [深度学习ctr](#深度学习ctr)
- [GNN](#gnn)
- [RL](#rl)
- [压缩与部署](#压缩与部署)
- [架构](#架构)
    - [TensorRT](#tensorrt)
    - [开源gnn平台](#开源gnn平台)
- [课程资源](#课程资源)
    - [无监督](#无监督)
    - [tf2.0](#tf20)
- [其他](#其他)

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

### 标准

[ChineseGLUE：为中文NLP模型定制的自然语言理解基准](https://mp.weixin.qq.com/s/14XQqFcLG1wMyB2tMABsCA)

===here

### bert变种

#### gated transformer-xl

[Stabilizing Transformers for Reinforcement Learning](https://arxiv.org/abs/1910.06764)

摘要：得益于预训练语言模型强大的能力，这些模型近来在 NLP 任务上取得了一系列的成功。这需要归功于使用了 transformer 架构。但是在强化学习领域，transformer 并没有表现出同样的能力。本文说明了为什么标准的 transformer 架构很难在强化学习中优化。研究者同时提出了一种架构，可以很好地提升 transformer 架构和变体的稳定性，并加速学习。研究者将提出的架构命名为 Gated Transformer-XL (GTrXL)，该架构可以超过 LSTM，在多任务学习 DMLab-30 基准上达到 SOTA 的水平。

推荐：本文是 DeepMind 的一篇论文，将强化学习和 Transformer 结合是一种新颖的方法，也许可以催生很多相关的交叉研究。

#### bert蒸馏、量化、剪枝

[BERT 瘦身之路：Distillation，Quantization，Pruning](https://mp.weixin.qq.com/s/ir3pLRtIaywsD94wf9npcA)

### 对话

#### convai

[GitHub超1.5万星NLP团队热播教程：使用迁移学习构建顶尖会话AI](https://mp.weixin.qq.com/s/lHzZjY98WxNeQDjTH7VXAw)

[https://convai.huggingface.co/](https://convai.huggingface.co/)

[https://github.com/huggingface/transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai)

## 语音

### 中文语音识别

[实战：基于tensorflow 的中文语音识别模型 | CSDN博文精选](https://mp.weixin.qq.com/s/rf6X5Iz4IOVtTdT8qVSi4Q)

## 视频

### 快手相关工作

[AI碰撞短视频，从推荐到直播，快手探索了这些ML新思路](https://mp.weixin.qq.com/s/Wn-5VD2-YWwVUWCMEy-lvw)

视频推荐、内容分发优化、视频码率优化这三方面探索提升快手视频体验的新方案。

#### EIUM：讲究根源的快手短视频推荐

[Explainable Interaction-driven User Modeling over Knowledge Graph for Sequential Recommendation](https://dl.acm.org/citation.cfm?id=3350893)

#### Comyco：基于质量感知的码率自适应策略

[Comyco: Quality-aware Adaptive Video Streaming via Imitation Learning](https://dl.acm.org/citation.cfm?id=3351014)

#### Livesmart：智能CDN调度

[Livesmart: a QoS-Guaranteed Cost-Minimum Framework of Viewer Scheduling for Crowdsourced Live Streaming](https://dl.acm.org/citation.cfm?id=3351013)

### google

[通过未标记视频进行跨模态时间表征学习](https://mp.weixin.qq.com/s/5qC70NoTBQ95vjI4cGl66g)

两篇：

[VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/abs/1904.01766)，VideoBert模型。

[Contrastive Bidirectional Transformer for Temporal Representation Learning](https://arxiv.org/abs/1906.05743)，CBT模型。

## 多目标

[阿里提出多目标优化全新算法框架，同时提升电商GMV和CTR](https://mp.weixin.qq.com/s/JXW--wzpaFwRHSSvZEA0mg)

## 推荐



## ctr

### 传统ctr

[https://daiwk.github.io/posts/dl-traditional-ctr-models.html](https://daiwk.github.io/posts/dl-traditional-ctr-models.html)

### 深度学习ctr

[https://daiwk.github.io/posts/dl-dl-ctr-models.html](https://daiwk.github.io/posts/dl-dl-ctr-models.html)

## GNN




## RL

## 压缩与部署

[GDP：Generalized Device Placement for Dataflow Graphs](https://arxiv.org/pdf/1910.01578.pdf)

大型神经网络的运行时间和可扩展性会受到部署设备的影响。随着神经网络架构和异构设备的复杂性增加，对于专家来说，寻找合适的部署设备尤其具有挑战性。现有的大部分自动设备部署方法是不可行的，因为部署需要很大的计算量，而且无法泛化到以前的图上。为了解决这些问题，研究者提出了一种高效的端到端方法。该方法基于一种可扩展的、在图神经网络上的序列注意力机制，并且可以迁移到新的图上。在不同的表征深度学习模型上，包括 Inception-v3、AmoebaNet、Transformer-XL 和 WaveNet，这种方法相比人工方法能够取得 16% 的提升，以及比之前的最好方法有 9.2% 的提升，在收敛速度上快了 15 倍。为了进一步减少计算消耗，研究者在一系列数据流图上预训练了一个策略网络，并使用 superposition 网络在每个单独的图上微调，在超过 50k 个节点的大型图上得到了 SOTA 性能表现，例如一个 8 层的 GNMT。

推荐：本文是谷歌大脑的一篇论文，通过图网络的方法帮助将模型部署在合适的设备上。推荐收到硬件设备限制，需要找到合适部署图的方法的读者参考。

[ICCV 2019 提前看 | 三篇论文，解读神经网络压缩](https://mp.weixin.qq.com/s/86A9kZkl_sQ1GrHMJ6NWpA)

+ [MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://arxiv.org/abs/1903.10258)

旷视。近年来，有研究表明无论是否保存了原始网络的权值，剪枝网络都可以达到一个和原始网络相同的准确率。因此，通道剪枝的本质是逐层的通道数量，也就是网络结构。鉴于此项研究，Metapruning 决定直接保留裁剪好的通道结构，区别于剪枝的裁剪哪些通道。

本文提出来一个 Meta network，名为 PruningNet，可以生成所有候选的剪枝网络的权重，并直接在验证集上评估，有效的搜索最佳结构。

+ [Data-Free Learning of Student Networks](https://arxiv.org/abs/1904.01186v1)

该篇论文是华为提出的一篇蒸馏方向的论文，其主要的创新点是提出的蒸馏过程不需要原始训练数据的参与。

+ [Correlation Congruence for Knowledge Distillation](https://arxiv.org/abs/1904.01802)

这篇论文是由商汤提出的一篇蒸馏方向论文，其主要的亮点在于研究样本之间的相关性，利用这种相关性作为蒸馏的知识输出。


## 架构

### TensorRT

[手把手教你采用基于TensorRT的BERT模型实现实时自然语言理解](https://mp.weixin.qq.com/s/rvoelBO-XjbswxQigH-6TQ)

### 开源gnn平台

[https://daiwk.github.io/posts/platform-gnn-frameworks.html](https://daiwk.github.io/posts/platform-gnn-frameworks.html)



## 课程资源

### 无监督

[14周无监督学习课程，UC伯克利出品，含课件、视频](https://mp.weixin.qq.com/s/leQEWqfBfLfAnyD0LU0uqA)

### tf2.0

[https://daiwk.github.io/posts/platform-tensorflow-2.0.html](https://daiwk.github.io/posts/platform-tensorflow-2.0.html)

[TensorFlow 2.0 常用模块1：Checkpoint](https://mp.weixin.qq.com/s/KTj3wJSA4_95pJvN-pkoZw)


## 其他

不知道是啥。。

nips18的tadam还有META-LEARNING WITH LATENT EMBEDDING OPTIMIZATION这篇 基本思路都是把问题先转化成做样本matching的deep metric learning任务 并对类目信息做task condition