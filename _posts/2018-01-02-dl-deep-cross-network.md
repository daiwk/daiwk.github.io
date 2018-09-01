---
layout: post
category: "dl"
title: "deep & cross network for ad click predictions"
tags: [deep & cross network, DCN]
---

目录

<!-- TOC -->

- [概述](#%E6%A6%82%E8%BF%B0)
- [related work](#related-work)
    - [embedding方法](#embedding%E6%96%B9%E6%B3%95)
        - [Factorization machines(FMs)](#factorization-machinesfms)
        - [Field-aware Factorization Machines(FFMs)](#field-aware-factorization-machinesffms)
    - [神经网络](#%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
- [贡献](#%E8%B4%A1%E7%8C%AE)
- [网络结构](#%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84)
    - [Embedding and Stacking Layer](#embedding-and-stacking-layer)
    - [Cross Network](#cross-network)
    - [Deep Network](#deep-network)
    - [Combination Layer](#combination-layer)

<!-- /TOC -->

## 概述

简称DCN。

论文地址：[deep & cross network for ad click predictions](https://arxiv.org/abs/1708.05123)

在ctr预估这种场景中，有大量的离散和类别特征，所以特征空间非常大且稀疏。线性模型，例如logistic regression，简单，有可解释性，易扩展性，而交叉特征可以显著增加模型的表示能力。但这些组合徨需要人工的特征工程，或者exhaustive searching，而且，产生unseen的特征interactions非常困难。

本文提出了cross network，可以使用自动的方式实现显式的特征交叉。cross network包括了多种层，其中最高度的interactions是由层的深度决定的。每一层会基于已有的部分产出高阶interactions，并保持前层的interactions。**本文实现了cross network和dnn的联合训练。dnn能捕捉特征间非常复杂的interactions，但相比cross network而言，需要更多参数（几乎差一个数量级？an order of），而且难以显式地产出交叉特征，并且可能难以高效地学习特征的interactions。**

## related work

为了避免extensive task-specific特征工程，主要有两类方法：embedding方法和神经网络。

### embedding方法

#### Factorization machines(FMs)

将离散特征映射到低维的稠密vector，然后通过vector的内积学习特征交互。

#### Field-aware Factorization Machines(FFMs)

让每个feature可以学习更多的vectors，而每个vector与一个field有关。

### 神经网络

如果有足够的隐层及隐单元，DNN能够近似任意函数。而现实问题所对应的函数往往不是『任意』的。

## 贡献

+ 在cross network中，每一层都有feature crossing，能够学习交叉特征，并不需要人为设计的特征工程。
+ cross network简单有效。多项式度（polynomial degree）随着网络层数增加而增加
+ 十分节约内存，且易于使用
+ DCN相比于其他模型有更出色的效果，与DNN模型相比，较少的参数却取得了较好的效果。

## 网络结构

<html>
<br/>
<img src='../assets/deepcross.jpg' style='max-height: 350px'/>
<br/>
</html>

### Embedding and Stacking Layer

将sparse特征（例如类别型的特征，输入就是一个二进制的one-hot，如[0,1,0,0]）embedding成稠密向量：

`\[
x_{embed,i}=W_{embed,i}x_i
\]`

其中，`\(x_{embed,i}\)`是embedding后的向量，`\(x_i\)`是第i个类目的二进制输入，`\(W_{embed,i}\in R^{n_e\times n_v}\)`是一个embedding矩阵，`\(n_e\)`是embedding的size，`\(n_v\)`是词典的size。

然后将embedding向量和归一化后的dense特征拼接成一个向量：

`\[
x_0=[x^T_{embed,1},...x^T_{embed,k},x^T_{dense}]
\]`



### Cross Network

### Deep Network

### Combination Layer


