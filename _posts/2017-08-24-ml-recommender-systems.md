---
layout: post
category: "ml"
title: "推荐系统"
tags: [svd, svd++, als, rbm-cf, fm, eALS, ]
---

目录

<!-- TOC -->

- [eALS](#eals)
  - [简介](#%E7%AE%80%E4%BB%8B)

<!-- /TOC -->

[从item-base到svd再到rbm，多种Collaborative Filtering(协同过滤算法)从原理到实现](http://blog.csdn.net/dark_scope/article/details/17228643)

附：简单说一下svd++，就是加上一个user bias，再加一个item bias，而user向量再加上这个用户的邻域信息(图中的y是用户的N(u)个历史item的隐式反馈)：

`\[
\hat{r_{u i}}=\mu+b_{i}+b_{u}+\left(p_{u}+\frac{1}{\sqrt{|N(u)|}} \sum_{i \in N(u)} y_{i}\right) q_{i}^{T}
\]`

## eALS

[Fast Matrix Factorization for Online Recommendation with Implicit Feedback](https://arxiv.org/pdf/1708.05024.pdf)

### 简介

以往的MF模型对于missing data，都是直接使用uniform weight(均匀分布)。然而在真实场景下，这个均匀分布的假设往往是不成立的。而且很多offline表现好的，到了动态变化的online场景上，往往表现不好。

文章使用**item的popularity**来给missing data权重，并提出了element-wise Alternating Least Squares(**eALS**)，来对missing data的权重是变量的问题进行学习。对于新的feedback，设计了增量更新的策略。对于两个offline和online的公开数据集，eALS都能取得比sota的隐式MF方法更好的效果。

之前的MF大多关注显式反馈，也就是用户的打分行为直接表达了用户对item的喜好程度。这种建模方式是基于这样一种假设：**大量的无标注的ratings(例如，missing data)与用户的preference无关**。这就大大减轻了建模的工作量，大量类似的复杂模型被提出了，例如SVD++、time-SVD等。

但在实际应用中，用户往往有大量的隐式反馈，例如浏览历史、购买历史等，而负反馈却是稀缺的。如果只对正反馈建模，那得到的是用户profile的一种biased的表示。针对这种负反馈缺失的场景（也叫one-class problem，参考[One-class collaborative filtering](xx)），比较流行的解法是把missing data当做负反馈。但如果观测数据和missing data都要考虑的话，算法的学习效率就会大大降低。

