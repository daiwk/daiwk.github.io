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
        - [复杂度分析](#%E5%A4%8D%E6%9D%82%E5%BA%A6%E5%88%86%E6%9E%90)
    - [Deep Network](#deep-network)
        - [复杂度分析](#%E5%A4%8D%E6%9D%82%E5%BA%A6%E5%88%86%E6%9E%90)
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
<img src='../assets/deepcross.png' style='max-height: 350px'/>
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

显式地进行feature crossing，每一个cross layer的公式如下：

`\[
x_{l+1}=x_0x^T_lw_l+b_l+x_l=f(x_l,w_l,b_l)+x_l
\]`

其中，

+ `\(x_l,x_{l+1}\in R^d\)`是第l和第l+1层的cross layer输出的列向量
+ `\(w_l,b_l\in R^d\)`是第l层的权重和bias
+ 每一层的cross layer在进行完feature crossing的`\(f\)`之后，又把input加回来
+ 借鉴residual network的思想，映射函数`\(f:R^d \mapsto R^d\)`拟合残差`\(x_{l+1}-x{l}\)`

<html>
<br/>
<img src='../assets/deep-cross-cross-layer.png' style='max-height: 150px'/>
<br/>
</html>

可见，维数分别是dx1,1xd,dx1，最终输出得到的就是一个dx1的向量。所以第二项需要对上一层的输出转置成1xd。

相当于考虑了从1到l+1阶的所有特征组合。

#### 复杂度分析

假设cross layer有`\(L_c\)`层，`\(d\)`是input的维数，那么，整个cross network的参数数量就是：

`\[
d\times L_c \times 2
\]`

这里乘了2，因为有w和b两个参数，每个参数在每一层里都是d维。时间和空间复杂度都是线性的。

### Deep Network

dnn部分就是正常的dnn：

`\[
h_{l+1}=f(W_lh_l+b_l)
\]`

+ `\(h_l \in R^{n_l},h_{l+1}\in R^{n_{l+1}}\)`
+ `\(W_l\in R^{n_{l+1}\times n_l},b_l\in R^{n_{l+1}}\)`
+ `\(f\)`是ReLU

#### 复杂度分析

假设所有层的size一样大，都是`\(m\)`。假设层数是`\(L_d\)`，总参数数量为：

`\[
d\times m+m+(m^2+m)\times (L_d-1)
\]`

其中，第一层是和embedding的`\(x_0\)`相连的，所以是`\(d\times m+m\)`

### Combination Layer

combination layer如下：

`\[
p=\sigma ([x^T_{L_1},h^T_{L_2}]w_{logits})
\]`

其中，`\(x_{L_1}\in R^d\)`是cross network的输出，`\(h_{L_2}\in R^m\)`是dnn的输出。`\(w_{logits}\in R^{(d+m)}\)`是权重，`\(\sigma (x)=1/(1+exp(-x))\)`。

损失函数是log loss加上l2正则：

`\[
loss=-\frac{1}{N}\sum ^N_{i=1}y_ilog(p_i)+(1-y_i)log(1-p_i)+\lambda \sum _l\parallel w_l \parallel ^2
\]`

