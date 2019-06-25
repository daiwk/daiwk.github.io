---
layout: post
category: "cv"
title: "hekaiming tutorials"
tags: [cvpr2017, hekaiming, Learning Deep Features
for Visual Recognition, eccv2018]
---

目录

<!-- TOC -->

- [cvpr2017](#cvpr2017)
  - [Convolutional Neural Networks: Recap](#Convolutional-Neural-Networks-Recap)
  - [ResNet](#ResNet)
  - [ResNeXt](#ResNeXt)
- [eccv2018](#eccv2018)

<!-- /TOC -->

## cvpr2017

cvpr2017 hekaiming的Learning Deep Features for Visual Recognition

[Learning Deep Features for Visual Recognition](http://deeplearning.csail.mit.edu/cvpr2017_tutorial_kaiminghe.pdf)

### Convolutional Neural Networks: Recap

### ResNet

### ResNeXt

参考[何恺明团队新作ResNext：Instagram图片预训练，挑战ImageNet新精度](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652047873&idx=3&sn=4dc44a7b4f050a6e55734faeb4297929&chksm=f1207ef0c657f7e61b4b14bfd2d3f915bf5fe79ffbd3dfd61a6fa23f18bd2179025e8f635fb4&mpshare=1&scene=1&srcid=0625VZIm3H5Kd9PJYDeAExh0&pass_ticket=b29oFqpJS3l2z9rkgAH2HsO1MhYXnw6w%2FmAI30o3T46KiQgaX30qaNd4uUXfW4zq#rd)

ResNeXt-101：[Exploring the Limits of Weakly Supervised Pretraining](https://arxiv.org/pdf/1805.00932.pdf)

如果没有有监督式预训练，很多方法现在还被认为是一种蛮干 ImageNet数据集实际上是预训练数据集。我们现在实际上对数据集的预训练了解相对较少。其原因很多：比如现存的预训练数据集数量很少，构建新数据集是劳动密集型的工作，需要大量的计算资源来进行实验。然而，鉴于预训练过程在机器学习相关领域的核心作用，扩大我们在这一领域的科学知识是非常重要的。

本文试图通过研究一个未开发的数据体系来解决这个复杂的问题：使用外部社交媒体上数十亿的带有标签的图像作为数据源。该数据源具有大而且不断增长的优点，而且是“免费”注释的，因为数据不需要手动标记。显而易见，对这些数据的训练将产生良好的迁移学习结果。

本文的主要成果是，在不使用手动数据集管理或复杂的数据清理的情况下，利用数千个不同主题标签作为标记的数十亿幅Instagram图像进行训练的模型，表现出了优异的传输学习性能。在目标检测和图像分类任务上实现了对当前SOTA性能的提升。在ImageNet-1k图像分类数据集上获得single-crop 最高准确率达到了85.4%，AP达到了45.2%。当在ImageNet-1k上训练（或预训练）相同模型时，分数分别为79.8％和43.7％。然而，我们的主要目标是提供关于此前未开发的制度的新实验数据。为此，我们进行了大量实验，揭示了一些有趣的趋势。

代码：[https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/)

## eccv2018

[http://kaiminghe.com/eccv18tutorial/eccv2018_tutorial_kaiminghe.pdf](http://kaiminghe.com/eccv18tutorial/eccv2018_tutorial_kaiminghe.pdf)

