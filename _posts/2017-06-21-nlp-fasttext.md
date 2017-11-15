---
layout: post
category: "nlp"
title: "fasttext"
tags: [fasttext, ]
---

目录
<!-- TOC -->

- [1. bin使用方法](#1-bin使用方法)
    - [训练](#训练)
    - [预测](#预测)
- [2. python使用方法](#2-python使用方法)
    - [安装](#安装)

<!-- /TOC -->


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

