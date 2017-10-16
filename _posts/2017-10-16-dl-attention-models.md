---
layout: post
category: "dl"
title: "从2017年顶会论文看Attention Model"
tags: [attention models, ]
---

参考[从2017年顶会论文看Attention Model - PaperWeekly 第50期](https://mp.weixin.qq.com/s/R59A_qviCMjklIlmTE-qww)


* [引言](#引言)
  * [单层Attention Model](#单层attention-model)
  * [多注意力机制（Hierarchical Attention &amp; Dual Attention）](#多注意力机制hierarchical-attention--dual-attention)
        * [基于知识图谱或者领域知识的注意力机制（Knowledge-base Attention）](#基于知识图谱或者领域知识的注意力机制knowledge-base-attention)
  * [参考文献](#参考文献)

## 引言

Attention Model 在 Image Caption、Machine Translation、Speech Recognition 等领域上有着不错的结果。那么什么是 Attention Model 呢？

举个例子，给下面这张图片加字幕（Image Caption）：一只黄色的小猫带着一个鹿角帽子趴在沙发上。可以发现在翻译的过程中我们的注意力由小猫到鹿角帽子再到沙发（小猫→鹿角帽子→沙发）。其实在很多和时序有关的事情上，人类的注意力都不是一成不变的，随着事情（时间）发展，我们的注意力不断改变。 

这篇文章的预备知识是 Decoder-Encoder 模型。本文主要做一个介绍，基本不会有公式推导，旨在让大家对 Attention Model 的变型和应用有一个大概的印象。


## 单层Attention Model

2014 年发表的文章 Neural Machine Translation by Jointly Learning to Align and Translate 使用了单层 Attention Model 解决机器翻译中不同长度的源语言对齐问题。使用 Attention Model 的基本思想是目标语言端的词往往只与源语言端部分词相关。而这个相关度通常用概率的形式表示。

这个过程基本为：首先计算当前上下文的环境与每个源语言词语的相关度（一个得分），然后使用 softmax 公式，将这个相关度转化为概率的形式，最后用得到的概率乘以对应源语言端词的隐含表示作为该词对预测目标的贡献，将所有源语言端的词贡献加起来作为预测下一个词的部分输入。

其中计算上下文环境与源语言词语的相关得分，是根据语言特性设计的一个对齐模型（Alignment Model）。通常情况下，单层 Attention Model 的不同之处主要在于相关分数计算方式的不同，接下来我们介绍三种通用的计算方式。同时在后文中，不再重复叙述 Attention Model 中根据相关分数计算输出向量的过程。

<html>
<br/>

<img src='../assets/single-attention.png' style='max-height: 350px'/>
<br/>

</html>

论文 Dipole: Diagnosis Prediction in Healthcare via Attention-based Bidirectional Recurrent Neural Networks，介绍了单个 Attention Model 在医疗诊断预测中的应用。

这个模型的输入是用户前 t 次的医疗代码（每次的医疗代码用 one-hot 的形式表示），输出是用户下一时刻的医疗诊断类型。使用 Attention Model 的思想是：用户下一时刻被诊断的疾病类型可能更与前面某一次或某几次的医疗诊断相关。论文模型框架如下。 

<html>
<br/>

<img src='../assets/attention-bi-rnn.jpg' style='max-height: 350px'/>
<br/>

</html>

论文 Dynamic Attention Deep Model for Article Recommendation by Learning Human Editors’ Demonstration 介绍了单个 Attention Model 在新闻推荐/筛选领域的应用。该模型的输入是一个文章的文本和种类信息，输出是0/1，表示输入的新闻是否被选中（二分类问题）。

下图展示的是该模型的 Attention Model 部分，未展示的部分是处理输入数据的过程，该过程是通过 CNN 等模型将文本和种类特征处理成固定维度的隐含向量表示。


## 多注意力机制（Hierarchical Attention & Dual Attention）

## 基于知识图谱或者领域知识的注意力机制（Knowledge-base Attention）


## 参考文献

+ KDD-2017 
[1] Dipole: Diagnosis Prediction in Healthcare via Attention-based Bidirectional Recurrent Neural Networks
[2] A Context-aware Attention Network for Interactive Interactive Question Answering
[3] Dynamic Attention Deep Model for Article Recommendation by Learning Human Editors’ Demonstration
[4] GRAM: Graph-based Attention Model For Healthcare Representation Learning
[5] Learning to Generate Rock Descriptions from Multivariate Well Logs with Hierarchical Attention

+ SIGIR-2017 
[6] Enhancing Recurrent Neural Networks with Positional Attention for Question Answering
[7] Attentive Collaborative Filtering: Multimedia Recommendation with Item- and Component-Level Attention
[8] Video Question Answering via Attribute-Augmented Attention Network Learning
[9] Leveraging Contextual Sentence Relations for Extractive Summarization Using a Neural Attention Model

+ Recsys-2017 
[10] Interpretable Convolutional Neural Networks with Dual Local and Global Attention for Review Rating Prediction
