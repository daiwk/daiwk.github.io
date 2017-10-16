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
