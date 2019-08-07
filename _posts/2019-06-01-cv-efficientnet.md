---
layout: post
category: "cv"
title: "EfficientNet"
tags: [EfficientNet, GPipe, ]
---

目录

<!-- TOC -->

- [efficientnet](#efficientnet)
- [改进版](#改进版)

<!-- /TOC -->

## efficientnet

参考[谷歌出品EfficientNet：比现有卷积网络小84倍，比GPipe快6.1倍](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652046349&idx=4&sn=721193942579e26cbc65e026db548674&chksm=f12070fcc657f9eada2bf6f1ccd7ed199b36380d79ce3bb11c916a92491ee0c6cd3daf184954&mpshare=1&scene=1&srcid=&pass_ticket=TloMdmvUbLd5jnKvVTzrccQhGuskwL6KQ0HhJLF56Nwtcb16%2BVvMA09bw32tFrjs#rd)

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

[https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

目前提高CNN精度的方法，主要是通过任意增加CNN**深度**或**宽度**，或使用更大的输入图像分辨率进行训练和评估。

以固定的资源成本开发，然后按比例放大，以便在获得更多资源时实现更好的准确性。例如ResNet可以通过增加层数从ResNet-18扩展到ResNet-200。

劣势就是，往往需要进行繁琐的微调。一点点的摸黑去试、还经常的徒劳无功

作者发现只要对网络的深度、宽度和分辨率进行合理地平衡，就能带来更好的性能。基于这一观察，科学家提出了一种新的缩放方法，使用简单但高效的复合系数均匀地缩放深度、宽度和分辨率的所有尺寸。

+ 第一步是执行网格搜索，在固定资源约束下找到基线网络的不同缩放维度之间的关系（例如，2倍FLOPS），这样做的目的是为了找出每个维度的适当缩放系数。
+ 然后应用这些系数，将基线网络扩展到所需的目标模型大小或算力预算。

与传统的缩放方法相比，这种复合缩放方法可以持续提高扩展模型的准确性和效率，和传统方法对比结果：MobileNet（+ 1.4％ imagenet精度），ResNet（+ 0.7％）。

新模型缩放的有效性，很大程度上也依赖基线网络。

为了进一步提高性能，研究团队还通过使用AutoML MNAS框架执行神经架构搜索来开发新的基线网络，该框架优化了准确性和效率（FLOPS）。 

由此产生的架构使用移动倒置瓶颈卷积（MBConv），类似于MobileNetV2和MnasNet，但由于FLOP预算增加而略大。然后，通过扩展基线网络以获得一系列模型，被称为EfficientNets。

## 改进版

[AutoML构建加速器优化模型首尝试，谷歌发布EfficientNet-EdgeTPU](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650767463&idx=3&sn=d37de5f41347916496881d10bf76a8be&chksm=871a4419b06dcd0f8c4119dfeed6a93afcc7a3a925905945ad058bbc51368e063cf01f2c05f9&scene=0&xtrack=1&pass_ticket=Kz97uXi0CH4ceADUC3ocCNkjZjy%2B0DTtVYOM7n%2FmWttTt5YKTC2DQT9lqCel7dDR#rd)

[https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu)

[https://coral.withgoogle.com/docs/](https://coral.withgoogle.com/docs/)

原文：[https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html](https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html)
