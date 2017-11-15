---
layout: post
category: "platform"
title: "Gluon"
tags: [Gluon, ]
---

目录


<!-- TOC -->

- [接口更简洁](#接口更简洁)
- [速度更快](#速度更快)
- [即是文档，又是教材](#即是文档又是教材)
- [Gluon与其他框架的对比](#gluon与其他框架的对比)
- [一系列链接](#一系列链接)

<!-- /TOC -->

参考[https://zhuanlan.zhihu.com/p/28648399?from=timeline&isappinstalled=0](https://zhuanlan.zhihu.com/p/28648399?from=timeline&isappinstalled=0)

## 接口更简洁

Gluon采用Keras和Numpy风格API，并且Layer可以自动判断输入长度。用过Chainer和Pytorch的人想必都体会过每一层都要记住前一层输出长度的麻烦，从卷积层到全联接层过度时长度计算更是痛苦，往往要运行一遍才知道。在Gluon里则没有这种问题，每层只要指定输出长度，输入长度则可以由系统自动计算。

## 速度更快

深度学习框架大体分为两类：以TensorFlow，caffe2为代表的静态图（Symbolic）框架和以Chainer，Pytorch为代表的动态图（Imperative）框架。静态图的优势在于速度快，省内存，便于线上部署。而动态图框架的优势是灵活，易用，debug方便，特别是在自然语言处理和增强学习等领域，比起静态图框架有显著优势。

Gluon同时支持灵活的动态图和高效的静态图，让你在享受动态编程的灵活易用的同时最小化性能的损失。而Gluon的HybridBlock和hybridize接口让你可以在静态动态间一键切换。0.11版Gluon比0.20版Pytorch快10%以上，在未来的一两个月我们会加入更多优化，再提高10%以上的性能。

## 即是文档，又是教材

Gluon教程包括深度学习理论讲解和代码实践。前五章每个例子都包括了两个版本。从零开始（from scratch）版本深入讲解所有细节，Gluon版本则着重演示高级封装的灵活高效。建议刚开始接触深度学习的同学从头开始顺序阅读，而已经有一定经验的同学可以跳过基础教程只看Gluon版。这套教程现在在Github上公开写作，共计划18章，已经完成了前五章。印刷出版和中文翻译也在计划中。

## Gluon与其他框架的对比

Tensorflow：Gluon同时支持静态图和动态图，在灵活性和速度上都有优势。但由于Gluon刚刚面市，在成熟度和线上部署方便还有不足。总的来说在做深度学习研究的同学不妨一试。

Pytorch：Gluon与Pytorch的相似度很高，而Gluon独特的静、动态图混合功能可以在不牺牲灵活性的前提下提高性能。如果你喜欢pytorch的简单易用又在乎性能，那么强烈建议你试一试Gluon。

## 一系列链接

+ 文档+教材：[https://github.com/zackchase/mxnet-the-straight-dope](https://github.com/zackchase/mxnet-the-straight-dope)

+ Gluon教程：[http://gluon.mxnet.io/](http://gluon.mxnet.io/) 

+ 安装教程：[https://mxnet.incubator.apache.org/versions/master/get_started/install.html](https://mxnet.incubator.apache.org/versions/master/get_started/install.html)

+ mxnet官网：[https://mxnet.incubator.apache.org/](https://mxnet.incubator.apache.org/)
