---
layout: post
category: "ml"
title: "few-shot learning"
tags: [few-shot, fewshot, 小样本, ]

---

目录

<!-- TOC -->

- [定义](#定义)
- [分类](#分类)
    - [model based](#model-based)
    - [metric based](#metric-based)
    - [optimization based](#optimization-based)

<!-- /TOC -->

参考[小样本学习（Few-shot Learning）综述](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650759590&idx=3&sn=d7573d59fdffae5fc7bb83ea77a26fa6&chksm=871aa5d8b06d2cce197f93547df18d412c173dcb2149f31797c1c9fbea2ec7948f6b1100ab3e&mpshare=1&scene=1&srcid=0429y1qxBdLrdZ8C5KgRqP0e&pass_ticket=ZltpB2rBQ1hXYEBIClPX5yEh187BJtWG0mhs8mqho%2F%2FHR%2BUZZqkJ8efTvcnsT5KF#rd)

参考[当小样本遇上机器学习 fewshot learning](https://blog.csdn.net/qq_16234613/article/details/79902085)

参考[Few-shot Learning: A Survey](https://arxiv.org/pdf/1904.05046.pdf)

关于meta learning，可以参考[https://daiwk.github.io/posts/dl-meta-learning.html](https://daiwk.github.io/posts/dl-meta-learning.html)


## 定义

Few-shot Learning 是 Meta Learning 在**监督学习**领域的应用

Meta Learning，又称为learning to learn，在meta training阶段将数据集分解为不同的**meta task**，去学习类别变化的情况下模型的泛化能力，在**meta testing**阶段，面对全新的类别，**不需要变动已有的模型**，就可以完成分类。

few-shot的训练集中包含了很多的类别，每个类别中有多个样本。

+ 在训练阶段，会在训练集中随机**抽取C个类别**，每个类别**K个样本**（总共CK个数据），构建一个 meta-task，作为模型的**支撑集（support set）**输入；
+ 再从这**C个类**中**剩余的数据**中抽取一批（batch）样本作为模型的**预测对象（batch set）**。

即要求模型从C\*K个数据中学会如何区分这C个类别，这样的任务被称为**C-way K-shot**(C类，每个类K个样本)问题。

训练过程中，每次训练 **(episode)**都会**采样**得到**不同meta-task**。这种机制使得模型学会**不同meta-task**中的**共性部分**，比如如何提取重要特征及比较样本相似等，**忘掉** meta-task中**task相关**部分，因此，在面对新的**未见过的 meta-task**时，也能较好地进行分类。

## 分类

主要分为Mode Based，Metric Based 和 Optimization Based三大类。

### model based

通过模型结构的设计快速在**少量样本上更新参数**，直接建立输入`\(x\)`和预测值`\(P\)`的映射函数。

`\[
P_{\theta}(y | x, S)=f_{\theta}(x, S)
\]`

[One-shot learning with memory-augmented neural networks](https://arxiv.org/abs/1605.06065)使用**记忆增强**的方法。基于记忆的神经网络方法早在2001年被证明可以用于meta-learning。通过**权重更新**来**调节bias**，并且通过学习**将表达**快速**缓存**到**记忆**中来调节输出。

利用循环神经网络的**内部记忆**单元**无法扩展**到需要对**大量新信息**进行编码的**新任务**上。因此，需要让存储在记忆中的表达既要**稳定**又要是**元素粒度访问**的，前者是说当**需要时就能**可靠地访问，后者是说可**选择性地**访问相关的信息；另外，参数数量**不能被内存的大小束缚**。**神经图灵机**（NTMs）和**记忆网络**就符合这种必要条件。

参考[https://daiwk.github.io/posts/dl-ntm-memory-networks.html](https://daiwk.github.io/posts/dl-ntm-memory-networks.html)



### metric based

通过度量**batch集中**的样本和**support集**中样本的**距离**，借助最近邻的思想完成分类

`\[
P_{\theta}(y | x, S)=\sum_{\left(x_{i}, y_{i}\right) \in S} k_{\theta}\left(x, x_{i}, S\right) y_{i}
\]`

### optimization based

认为**普通的梯度下降**方法**难以**在few-shot场景下**拟合**，因此通过调整优化方法来完成小样本分类的任务。

`\[
P_{\theta}(y | x, S)=f_{\theta(S)}(x)
\]`