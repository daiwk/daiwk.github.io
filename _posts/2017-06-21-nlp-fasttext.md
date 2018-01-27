---
layout: post
category: "nlp"
title: "fasttext"
tags: [fasttext, ]
---

目录
<!-- TOC -->

- [0. 原理](#0-原理)
    - [0.1 softmax回归](#01-softmax回归)
    - [0.2 分层softmax](#02-分层softmax)
- [1. bin使用方法](#1-bin使用方法)
    - [训练](#训练)
    - [预测](#预测)
- [2. python使用方法](#2-python使用方法)
    - [安装](#安装)

<!-- /TOC -->

## 0. 原理

参考[fastText原理及实践](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650736673&idx=4&sn=d5cb11250b28912accbc08ddb5d9c97b&chksm=871acc5fb06d45492ee54f3ff42e767bdc668d12c615b8ddc7e0aaeae7748aafe1aa53686176&mpshare=1&scene=1&srcid=0126lb9yqb0yUzJ30cmJc7ML&pass_ticket=5bhFv%2FwprJeuXSNRdbTSRrHitcKLawmckNnlQIBt%2FjavQ3ytUGB53qdfRz7NsZP4#rd)

在文本分类任务中，fastText（浅层网络）往往能取得和深度网络相媲美的精度，却在训练时间上比深度网络快许多数量级。在标准的多核CPU上， 能够训练**10亿词级别语料库**的词向量在**10分钟之内**，能够分类有着**30万多类别**的**50多万句子**在**1分钟**之内。

### 0.1 softmax回归

softmax回归被称作多项逻辑回归（multinomial logistic regression），是逻辑回归在处理多类别任务上的推广。

<html>
<br/>
<img src='../assets/fasttext-softmax.png' style='max-height: 300px'/>
<br/>
</html>

<html>
<br/>
<img src='../assets/fasttext-softmax-lr.png' style='max-height: 200px'/>
<br/>
</html>

### 0.2 分层softmax

softmax中，我们需要对所有的K个概率做归一化，这在|y|很大时非常耗时。分层softmax的基本思想是使用树的层级结构替代扁平化的标准Softmax，在计算`\(P(y=j)\)`时，只需计算**一条路径上的所有节点的概率值**，无需在意其它的节点。

<html>
<br/>
<img src='../assets/fasttext-hierachical-softmax.png' style='max-height: 100px'/>
<br/>
</html>

树的结构是根据**类标的频数**构造的**霍夫曼树**。

+ **K**个不同的类标组成所有的**叶子**节点。
+ **K-1个内部节点**作为内部参数。
+ 从根节点到某个叶子节点经过的节点和边形成一条路径，路径长度被表示为`\(L(y_j)\)`。

`\[
p(y_j)=\prod _{l=1}^{L(y_j)-1}\sigma (\left \lfloor n(y_j,l+1)=LC(n(y_j,l)) \right \rfloor \cdot \theta _{n(y_j,l)} ^TX) 
\]`

其中，

+ `\(l\)`表示第几层（从1开始）；
+ `\(\sigma (\cdot )\)`表示sigmoid函数；
+ `\(LC(n)\)`表示n节点的左孩子；
+ `\(\left \lfloor \right \rfloor \)`是一个特殊的函数，定义如下：

`\[
\left \lfloor x \right \rfloor =\left\{\begin{matrix}
1, if x =true
\\ -1,otherwise
\end{matrix}\right.
\]`

+ `\(\theata _{n(y_j,l)}\)`是中间节点`\(n(y_j,l)\)`的参数

高亮的节点和边是**从根节点到`\(y_2\)`的路径**，路径长度`\(L(y_2)=3?\)`，

`\[
\\P(y_2)=P(n(y_2,1),left)P(n(y_2,2),left)P(n(y_2,3),right)
\\=\sigma (\theta _{n(y_2,1)}^TX) \cdot \sigma (\theta _{n(y_2,2)}^TX) \cdot \sigma (-\theta _{n(y_2,3)}^TX) 
\]`

于是，从根节点走到叶子节点`\(y_2\)`，其实是做了3次二分类的lr。通过分层的Softmax，计算复杂度一下从|K|降低到**log2|K|**。

## 1. bin使用方法

编译好的bin地址：[https://daiwk.github.io/assets/fasttext](https://daiwk.github.io/assets/fasttext)

### 训练

+ 训练数据demo：[https://daiwk.github.io/assets/train_demo_fasttext.txt](https://daiwk.github.io/assets/train_demo_fasttext.txt)

格式(空格分割)：

```
__label__xx w1 w2 w3 ...
```

+ 训练命令：

```
./fasttext supervised -input train_demo_fasttext.txt -output haha.model
```

高级参数：

```
-minn 1 -maxn 6: 不用切词，1-6直接n-gram
```

### 预测


+ 测试数据demo：[https://daiwk.github.io/assets/test_demo_fasttext.txt](https://daiwk.github.io/assets/test_demo_fasttext.txt)

格式(空格分割）：

```
__label__00 key w1 w2 w3 ...
```

其中，**key可以中间有\1等任何分隔符，但key里面不能有空格**

+ 预测命令

```
cat test_demo_fasttext.txt | ./fasttext predict-prob haha.model.bin - 
```

+ 预测输出

```
key __label__xx probability
```

## 2. python使用方法

### 安装

```
xxxx/pip install cython
xxxx/pip install fasttext
```

