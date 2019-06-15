---
layout: post
category: "ml"
title: "推荐系统"
tags: [svd, svd++, als, rbm-cf, fm]
---

目录

<!-- TOC -->


<!-- /TOC -->

[从item-base到svd再到rbm，多种Collaborative Filtering(协同过滤算法)从原理到实现](http://blog.csdn.net/dark_scope/article/details/17228643)

附：简单说一下svd++，就是加上一个user bias，再加一个item bias，而user向量再加上这个用户的邻域信息(图中的y是用户的N(u)个历史item的隐式反馈)：

`\[
\hat{r_{u i}}=\mu+b_{i}+b_{u}+\left(p_{u}+\frac{1}{\sqrt{|N(u)|}} \sum_{i \in N(u)} y_{i}\right) q_{i}^{T}
\]`
