---
layout: post
category: "platform"
title: "dlrm"
tags: [dlrm, facebook,  ]
---

目录

<!-- TOC -->


<!-- /TOC -->

参考[想知道Facebook怎样做推荐？FB开源深度学习推荐模型](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650765233&idx=4&sn=3542b0b86ec78a81fef75feb39f2dfac&chksm=871ab3cfb06d3ad9999ed019ae623fff59db8ea77881bfc908f3d41432e8d7c9f0d56b7b58c6&scene=0&xtrack=1&pass_ticket=g0RhlU91yTm4YwdL6HxxS6fDU%2FNvWsf8uqd5BGk9%2Fewn4u2UU5gMclDp6uVTk%2Bm3#rd)

[https://github.com/facebookresearch/dlrm](https://github.com/facebookresearch/dlrm)

[Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091)

原始博客：[https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/)

DLRM模型有两大类特征：连续（dense）特征和类别（sparse）特征。使用emb处理类别特征，使用下方的多层感知机（MLP）处理连续特征。然后显式地计算不同特征的二阶交互（second-order interaction）。最后，使用顶部的多层感知机处理结果，并输入sigmoid函数中，得出点击的概率。

<html>
<br/>
<img src='../assets/dlrm.gif' style='max-height: 200px'/>
<br/>
</html>

