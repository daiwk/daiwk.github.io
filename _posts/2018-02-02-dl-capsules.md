---
layout: post
category: "dl"
title: "capsule"
tags: [capsules, capsule, ]
---

目录

<!-- TOC -->

- [1. cnn-challenges](#1-cnn-challenges)
- [2. equivariance](#2-equivariance)
- [3. capsule](#3-capsule)
- [4. dynamic-routing](#4-dynamic-routing)
    - [4.1 intuition](#41-intuition)
    - [4.2 calculating-a-capsule-output](#42-calculating-a-capsule-output)
- [5. iterative-dynamic-routing](#5-iterative-dynamic-routing)
- [6. max-pooling-shortcoming](#6-max-pooling-shortcoming)
- [7. significant-of-routing-by-agreement-with-capsules](#7-significant-of-routing-by-agreement-with-capsules)
- [8. capsnet-architecture](#8-capsnet-architecture)
- [9. loss-function-margin-loss](#9-loss-function-margin-loss)
- [10. capsnet-model](#10-capsnet-model)
    - [10.1 primarycapsules](#101-primarycapsules)
    - [10.2 squash-function](#102-squash-function)
    - [10.3 digicaps-with-dynamic-routing](#103-digicaps-with-dynamic-routing)
    - [10.4 image-reconstruction](#104-image-reconstruction)
    - [10.5 reconstruction-loss](#105-reconstruction-loss)
- [11. what-capsule-is-learning](#11-what-capsule-is-learning)

<!-- /TOC -->

**代码**：
[https://github.com/Sarasra/models/tree/master/research/capsules](https://github.com/Sarasra/models/tree/master/research/capsules)

paper:
[Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)

参考 [Hinton胶囊网络代码正式开源，5天GitHub fork超1.4万](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652012657&idx=1&sn=6e7c8bb25d41e7c73682fc6e539c0e1c&chksm=f121f480c6567d969b1e48f649665d8c5d5122559641b944f37535846c77f645e9f435189b3d&mpshare=1&scene=1&srcid=0201iw6QfDNUP6ejcN6Cp4L5&pass_ticket=qs99iMGY1jAAqC7Y%2F5bWFAKdSVWrNCDdPaLjWTEyZ612ZoU1cwNmbIOusMI23vOr#rd)


本文主要参考jhui的博客：
[https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/)

## 1. cnn-challenges

**神经元的激活水平**通常被解释为**检测特定特征**的可能性【例如，如果图中蓝色的比例很高，可能某个神经元的激活值就特别大】。

CNN善于**检测特征**，却在探索**特征（视角，大小，方位）之间**的**空间关系**方面效果较差。(一个简单的CNN模型可以正确提取鼻子、眼睛和嘴巴的特征，但如果一张图里，鼻子和眼睛错位了，或者一只眼睛倾斜了一定的角度，那这张图仍然有可能错误地激活神经元导致认为这张图就是人脸)

假设每个神经元都包含特征的**可能性**和**属性**【**这里就叫做胶囊(capsule)了，也就是说，里面包含的不是一个值(a single scaler value)，而是一个向量(vector)**】。例如，神经元输出的是一个包含**[可能性，方向，大小]**的向量。利用这种空间信息，就可以检测鼻子、眼睛和耳朵特征之间的方向和大小的一致性。此时，上面那张图对于人脸检测的激活输出就会低很多。

## 2. equivariance

为了CNN能够处理不同的视角或变体，我们添加了**更多**的神经元和层。尽管如此，这种方法倾向于记忆数据集，而不是得出一个比较通用的解决方案，它需要**大量的训练数据来覆盖不同的变体，并避免过拟合**。MNIST数据集包含55,000个训练数据，也即每个数字都有5,500个样本。但是，儿童看过几次就能记住数字。现有的包括CNN在内的深度学习模式在**利用数据方面效率十分低下**。

胶囊网络不是训练来捕捉特定变体的特征，而是捕捉特征及其变体的可能性。所以胶囊的目的不仅在于**检测特征**，还在于**训练模型**来**学习变体**。

这样，**相同的胶囊**就可以检测**不同方向**的**同一个物体类别**。

+ **Invariance**对应**特征检测**，特征是不变的。例如，**检测鼻子的神经元不管什么方向，都检测鼻子**。但是，神经元空间定向的损失最终会损害这种invariance模型的有效性。
+ **Equivariance**对应**变体检测**，也即可以相互转换的对象（例如检测不同方向的人脸）。直观地说，胶囊网络**检测到脸部旋转了20°**，而不是实现与旋转了20°的变体相匹配的脸。通过强制模型学习胶囊中的特征变体，我们可以**用较少的训练数据**，**更有效地推断可能的变体**。此外，也可以更有效地**防止对抗攻击**。

## 3. capsule

胶囊是一组神经元，不仅捕捉**特征的可能性**，还捕捉**具体特征的参数**。

## 4. dynamic-routing

### 4.1 intuition

### 4.2 calculating-a-capsule-output

## 5. iterative-dynamic-routing

## 6. max-pooling-shortcoming

## 7. significant-of-routing-by-agreement-with-capsules

## 8. capsnet-architecture

## 9. loss-function-margin-loss

## 10. capsnet-model

### 10.1 primarycapsules

### 10.2 squash-function

### 10.3 digicaps-with-dynamic-routing

### 10.4 image-reconstruction

### 10.5 reconstruction-loss

## 11. what-capsule-is-learning
