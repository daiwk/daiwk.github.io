---
layout: post
category: "dl"
title: "mixmatch"
tags: [mixmatch, 半监督, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

[MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)

本文来自 Google Research，这一方法综合了**自洽正则化（Consistency Regularization）**，**最小化熵（Entropy Minimization）**以及**传统正则化（Traditional Regularization）**，取三者之长，并补三者之短，提出了 MixMatch 这一方法。在 CIFAR10 上，仅仅使用 250 个标签数据就达到 11% 的错误率，远超其他主流方法。

参考[https://zhuanlan.zhihu.com/p/66281890](https://zhuanlan.zhihu.com/p/66281890)

参考[谷歌首席科学家：半监督学习的悄然革命](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652045372&idx=4&sn=780cb3cb33f7e30ef9de59d7ca1b6177&chksm=f12074cdc657fddb7fff685107b70883d40c42d43e2665a938599d524ebec1f11da7122b90a8&scene=4)

原始blog：[https://towardsdatascience.com/the-quiet-semi-supervised-revolution-edec1e9ad8c](https://towardsdatascience.com/the-quiet-semi-supervised-revolution-edec1e9ad8c)

[https://github.com/google-research/mixmatch](https://github.com/google-research/mixmatch)

