---
layout: post
category: "ml"
title: "crf v.s. softmax"
tags: [crf, softmax]
---

目录

<!-- TOC -->

- [softmax与crf对比](#softmax%E4%B8%8Ecrf%E5%AF%B9%E6%AF%94)
- [Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks](#optimal-hyperparameters-for-deep-lstm-networks-for-sequence-labeling-tasks)

<!-- /TOC -->


## softmax与crf对比

参考[https://www.jiqizhixin.com/articles/2018-05-23-3](https://www.jiqizhixin.com/articles/2018-05-23-3)

摘一句重点：

**逐帧 softmax 和 CRF 的根本不同：前者将序列标注看成是 n 个 k 分类问题，后者将序列标注看成是 1 个 k^n 分类问题。**


参考[https://blog.csdn.net/bobobe/article/details/80489303](https://blog.csdn.net/bobobe/article/details/80489303)

## Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks

[Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks](https://arxiv.org/abs/1707.06799)

