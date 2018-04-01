---
layout: post
toc: true
category: "links"
title: "【置顶】导引"
tags: [导引, navigation]
---

目录：

<!-- TOC -->

- [**1. Basic Knowledges**](#1-basic-knowledges)
    - [1.1 Machine Learning](#11-machine-learning)
        - [1.1.1 **树模型**](#111-树模型)
        - [1.1.2 **基础算法**](#112-基础算法)
        - [1.1.3 **统计学习方法**](#113-统计学习方法)
    - [1.2 Deep Learning](#12-deep-learning)
        - [1.2.1 CNN](#121-cnn)
        - [1.2.2 RNN](#122-rnn)
        - [1.2.3 GAN](#123-gan)
        - [1.2.4 Reinforcement Learning](#124-reinforcement-learning)
        - [1.2.5 PNN(Progressive Neural Network)连续神经网络](#125-pnnprogressive-neural-network连续神经网络)
        - [1.2.6 图卷积网络](#126-图卷积网络)
        - [1.2.7 copynet](#127-copynet)
- [**2. Useful Tools**](#2-useful-tools)
    - [2.1 Datasets](#21-datasets)
    - [2.2 pretrained models](#22-pretrained-models)
    - [2.3 Deep Learning Tools](#23-deep-learning-tools)
        - [2.3.1 mxnet](#231-mxnet)
        - [2.3.2 theano](#232-theano)
        - [2.3.3 torch](#233-torch)
        - [2.3.4 tensorflow](#234-tensorflow)
        - [2.3.5 docker](#235-docker)
        - [2.4 docker-images](#24-docker-images)
- [**3. Useful Courses && Speeches**](#3-useful-courses--speeches)
    - [3.1 Courses](#31-courses)
    - [3.2 Speeches](#32-speeches)
    - [3.3 经典学习资源](#33-经典学习资源)
- [4. **Applications**](#4-applications)
    - [4.1 NLP](#41-nlp)
        - [4.1.1 分词](#411-分词)
        - [4.1.2 Text Abstraction](#412-text-abstraction)
    - [4.2 Image Processing](#42-image-processing)
        - [4.2.1 image2txt](#421-image2txt)
    - [4.3 Collections](#43-collections)
        - [4.3.1 csdn深度学习代码专栏](#431-csdn深度学习代码专栏)
        - [4.3.2 chiristopher olah的博客](#432-chiristopher-olah的博客)
        - [4.3.3 激活函数系列](#433-激活函数系列)
        - [4.3.4 梯度下降算法系列](#434-梯度下降算法系列)

<!-- /TOC -->

# **1. Basic Knowledges**
 
## 1.1 Machine Learning
 
### 1.1.1 **树模型**
 
1. 【站外资源汇总】

[https://daiwk.github.io/posts/links-useful-links.html#111-%E6%A0%91%E6%A8%A1%E5%9E%8B](https://daiwk.github.io/posts/links-useful-links.html#111-%E6%A0%91%E6%A8%A1%E5%9E%8B) 

2. 【站内资源汇总】
暂无

### 1.1.2 **基础算法**

1. 【站外资源汇总】

[https://daiwk.github.io/posts/links-useful-links.html#112-%E5%9F%BA%E7%A1%80%E7%AE%97%E6%B3%95](https://daiwk.github.io/posts/links-useful-links.html#112-%E5%9F%BA%E7%A1%80%E7%AE%97%E6%B3%95)

2. 【站内资源汇总】
暂无

### 1.1.3 **统计学习方法**

1. 【站外资源汇总】

[https://daiwk.github.io/posts/links-useful-links.html#113-%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95](https://daiwk.github.io/posts/links-useful-links.html#113-%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95)

2. 【站内资源汇总】
暂无

## 1.2 Deep Learning 
 
### 1.2.1 CNN

1. 【站外资源汇总】

[https://daiwk.github.io/posts/links-useful-links.html#121-cnn](https://daiwk.github.io/posts/links-useful-links.html#121-cnn)

2. 【站内资源汇总】
暂无
 
### 1.2.2 RNN
 
1. 【站外资源汇总】

[https://daiwk.github.io/posts/links-useful-links.html#122-rnn](https://daiwk.github.io/posts/links-useful-links.html#122-rnn)

2. 【站内资源汇总】

+ 小合辑（从lstm到gru到双向多层到nmt到attention）:[https://daiwk.github.io/posts/nlp-nmt.html](https://daiwk.github.io/posts/nlp-nmt.html)
+ attention model集合：[https://daiwk.github.io/posts/dl-attention-models.html](https://daiwk.github.io/posts/dl-attention-models.html)
+ 只要attention的tensor2tensor：[https://daiwk.github.io/posts/platform-tensor-to-tensor.html](https://daiwk.github.io/posts/platform-tensor-to-tensor.html)
+ nested lstm: [https://daiwk.github.io/posts/dl-nested-lstm.html](https://daiwk.github.io/posts/dl-nested-lstm.html)
+ indRNN: [https://daiwk.github.io/posts/dl-indrnn.html](https://daiwk.github.io/posts/dl-indrnn.html)
 
注：最6的几个材料【暂时还没抄过来，手动滑稽】：

+ colah的博客：翻译版之一[https://www.leiphone.com/news/201701/UIlrDBnwiqoQUbqB.html](https://www.leiphone.com/news/201701/UIlrDBnwiqoQUbqB.html)，原文：[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
+  augmented-rnns：google大脑的研究员在博客中讲述了Neural Turing Machine、Attentional Interfaces、Adaptive Computation Time和Neural Programmers四大部分。[英文原文](http://distill.pub/2016/augmented-rnns/)；[新智元翻译版](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2651986905&idx=4&sn=dcfdeb7c92826c0603569d5a86025536&chksm=f1216f28c656e63e309d7c92fd06a1c67ac96ebea2a8c6f90169cd944876fb367a8bf819b4f4&mpshare=1&scene=1&srcid=1002ho2GSC2PTnFhFUio3EYj&pass_ticket=DoiMlYDlmCK%2FTS99n6JzBzzsHdN7QoyC81j%2BvUNHFkqqmuADrJsZlH0yXSTgpVEB#rd)；gitbub博客代码：[https://github.com/distillpub/post--augmented-rnns]
 
### 1.2.3 GAN
 
 1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 1.2.4 Reinforcement Learning

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 1.2.5 PNN(Progressive Neural Network)连续神经网络

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 1.2.6 图卷积网络

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 1.2.7 copynet

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
# **2. Useful Tools**
 
## 2.1 Datasets
 
1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
## 2.2 pretrained models

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
## 2.3 Deep Learning Tools
 
1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 2.3.1 mxnet

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 2.3.2 theano

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 2.3.3 torch

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 2.3.4 tensorflow

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 2.3.5 docker

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 2.4 docker-images

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
# **3. Useful Courses && Speeches**
 
## 3.1 Courses

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
## 3.2 Speeches

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
## 3.3 经典学习资源

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
# 4. **Applications**
 
## 4.1 NLP

### 4.1.1 分词

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 4.1.2 Text Abstraction

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
## 4.2 Image Processing

### 4.2.1 image2txt

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
## 4.3 Collections

### 4.3.1 csdn深度学习代码专栏

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 4.3.2 chiristopher olah的博客

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 4.3.3 激活函数系列

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 
### 4.3.4 梯度下降算法系列

1. 【站外资源汇总】


2. 【站内资源汇总】
暂无
 