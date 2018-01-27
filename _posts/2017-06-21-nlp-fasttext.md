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

