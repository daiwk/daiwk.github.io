---
layout: post
category: "platform"
title: "lingvo"
tags: [lingvo, ]
---

目录

<!-- TOC -->


<!-- /TOC -->


代码：[https://github.com/tensorflow/lingvo](https://github.com/tensorflow/lingvo)

论文：[Lingvo: a Modular and Scalable Framework for Sequence-to-Sequence Modeling](https://arxiv.org/abs/1902.08295)

Lingvo 是一个能够为协作式深度学习研究提供完整解决方案的 Tensorflow 框架，尤其关注序列到序列模型。Lingvo 模型由模块化构件组成，这些构件灵活且易于扩展，实验配置集中且可定制。分布式训练和量化推理直接在框架内得到支持，框架内包含大量 utilities、辅助函数和最新研究思想的现有实现。

设计原则如下：

+ 单个代码块应该精细且**模块化**，它们会使用相同的接口，同时也**容易扩展**；
+ 实验应该是共享的、可比较的、可复现的、可理解的和正确的；
+ 性能应该可以**高效地扩展到生产规模的数据集**，或拥有**数百个加速器的分布式训练系统**；
+ 当模型从**研究转向产品**时应该尽可能共享代码。

