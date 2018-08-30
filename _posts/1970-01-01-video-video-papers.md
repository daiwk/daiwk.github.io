---
layout: post
category: "video"
title: "视频相关paper"
tags: [video, 视频]
---

目录

<!-- TOC -->

- [视频分类](#%E8%A7%86%E9%A2%91%E5%88%86%E7%B1%BB)

<!-- /TOC -->

## 视频分类

AAAI 2018

[Multimodal Keyless Attention Fusion for Video Classification](http://iiis.tsinghua.edu.cn/~weblt/papers/multimodal-keyless-attention.pdf)

Multimodal Representation意思是多模式表示，在行为识别任务上，文章采用了视觉特征（Visual Features，包含RGB特征 和 flow features）；声学特征（Acoustic Feature）；前面两个特征都是针对时序，但是时序太长并不适合直接喂到LSTM，所以作者采用了分割的方法（Segment-Level Features），将得到的等长的Segment喂到LSTM。

优点：多模态信息融合方法优化: 两路model分别lstm+attention,再concat ,能得到多模态 互补识别效果,train model 效果优于early fusion/later fusion

CVPR 2018

[Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](http://cn.arxiv.org/abs/1711.09550)

多组attention参数有效引入diversity,能学习到不同的注意力模式

不同component用各自attention(通过线性变换映射到另一个空间，相当于加约束),再concat


temporal action proposal: actionness score（精彩不精彩，二分类），然后取出分布连续且值比较高的片段，然后找边界（boundary aware network），然后加一些多尺度(action pyramid network)


[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750v1)

利用膨胀3D卷积网络(I3D)将视频的帧间差值做处理,再采用CNN进行分类。


AdaIN

style transfer: AdaIN(自适应示例正则化)，两个目标：纹理一致性+语义一致性（类似GAN）
