---
layout: post
category: "nlp"
title: "nlp+gnn"
tags: [nlp gnn, hgat, ]
---

目录

<!-- TOC -->

- [HGAT](#hgat)

<!-- /TOC -->

### HGAT

[节后收心困难？这15篇论文，让你迅速找回学习状态](https://mp.weixin.qq.com/s/aaz-s87vorroyepNCd9-AA)

[Heterogeneous Graph Attention Network](https://arxiv.org/abs/1903.07293)

[Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification](http://shichuan.org/doc/74.pdf)

本文是北京邮电大学和南洋理工发表于 EMNLP 2019 的工作，文章创新性地将短文本分类问题建模为异质图（多种类型的节点和边）并提出一种端到端的异质图神经网络。短文本分类往往会遇到稀疏性问题（文本过短，无法提供足够的信息）。本文通过构图可以可以的丰富短文本之间的联系，进而较好的解决稀疏性问题。

另外，本文创新地提出一种异质图神经网络 HGAT。HGAT 分别从节点级别和类型级别聚合信息来更新节点表示。最后本文在 6 个数据集上做了大量的实验来验证 HGAT 的优越性。
