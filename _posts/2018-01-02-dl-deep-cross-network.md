---
layout: post
category: "dl"
title: "deep & cross network for ad click predictions"
tags: [deep & cross network, DCN]
---

目录

<!-- TOC -->

- [概述](#概述)
- [related work](#related-work)
    - [embedding方法](#embedding方法)
        - [Factorization machines(FMs)](#factorization-machinesfms)
        - [Field-aware Factorization Machines(FFMs)](#field-aware-factorization-machinesffms)
    - [神经网络](#神经网络)

<!-- /TOC -->

## 概述

论文地址：[deep & cross network for ad click predictions](https://arxiv.org/abs/1708.05123)

在ctr预估这种场景中，有大量的离散和类别特征，所以特征空间非常大且稀疏。线性模型，例如logistic regression，简单，有可解释性，易扩展性，而交叉特征可以显著增加模型的表示能力。但这些组合徨需要人工的特征工程，或者exhaustive searching，而且，产生unseen的特征interactions非常困难。

本文提出了cross network，可以使用自动的方式实现显式的特征交叉。cross network包括了多种层，其中最高度的interactions是由层的深度决定的。每一层会基于已有的部分产出高阶interactions，并保持前层的interactions。**本文实现了cross network和dnn的联合训练。dnn能捕捉特征间非常复杂的interactions，但相比cross network而言，需要更多参数（几乎差一个数量级？an order of），而且难以显式地产出交叉特征，并且可能难以高效地学习特征的interactions。**

## related work

为了避免extensive task-specific特征工程，主要有两类方法：embedding方法和神经网络。

### embedding方法

#### Factorization machines(FMs)

#### Field-aware Factorization Machines(FFMs)

### 神经网络

