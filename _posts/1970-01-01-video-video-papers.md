---
layout: post
category: "video"
title: "视频相关paper"
tags: [video, 视频]
---

目录

<!-- TOC -->

- [视频分类](#%e8%a7%86%e9%a2%91%e5%88%86%e7%b1%bb)
- [视频表示学习](#%e8%a7%86%e9%a2%91%e8%a1%a8%e7%a4%ba%e5%ad%a6%e4%b9%a0)
  - [Dense Predictive Coding](#dense-predictive-coding)

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

## 视频表示学习

### Dense Predictive Coding

[Video Representation Learning by Dense Predictive Coding](https://arxiv.org/abs/1909.04656)

近期来自 VGG 的高质量工作，因为没有在主会议发表所以没有引起大范围关注，但保持了一贯低调又实用的风格。本文提出了一种新型的自监督学习（self-supervised learning）方法 Dense Predictive Coding，学习视频的时空表征，在动作识别任务（UCF101 和 HMDB51 数据集）上获得了 state-of-the-art 的正确率，并且用无需标注的自监督学习方法在视频动作识别上达到了 ImageNet 预训练的正确率。

自监督学习是利用无标注的数据设计代理任务（proxy task），使网络从中学到有意义的数据表征。本文设计的代理任务是预测未来几秒的视频的特征，并且用对比损失（contrastive loss）使得预测的特征和实际的特征相似度高，却不必要完全相等。因为在像素级别（pixel-level）预测未来的帧容易受到大量随机干扰如光照强度、相机移动的影响，而在特征级别（feature-level）做回归（regression）则忽视了未来高层特征的不可预测性（如视频的未来发展存在多种可能）。

文中的设计促使网络学习高层语义特征，避免了网络拘泥于学习低层特征。作者在不带标注的 Kinetics400 上训练了自监督任务（Dense Predictive Coding），然后在 UCF101 和 HMDB51 上测试了网络所学权重在动作识别上的正确率。

Dense Predictive Coding 在 UCF101 数据集上获得了 75.7% 的 top1 正确率，超过了使用带标注的 ImageNet 预训练权重所获得的 73.0% 正确率。该研究结果证明了大规模自监督学习在视频分类上的有效性。
