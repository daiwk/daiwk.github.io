---
layout: post
category: "ml"
title: "online learning"
tags: [online learning, ftrl, ]

---

目录

<!-- TOC -->

- [xxx](#xxx)
- [adaptive regularization](#adaptive-regularization)

<!-- /TOC -->

找到两个综述：

[Online Learning: A Comprehensive Survey](https://arxiv.org/pdf/1802.02871.pdf)

[A Survey of Algorithms and Analysis for Adaptive Online Learning](http://www.jmlr.org/papers/volume18/14-428/14-428.pdf)

美团的[Online Learning算法理论与实践](https://tech.meituan.com/2016/04/21/online-learning.html)

FTRL可以参考[https://daiwk.github.io/posts/ml-ftrl.html](https://daiwk.github.io/posts/ml-ftrl.html)

## xxx

aaa

## adaptive regularization

[Learning recommender systems with adaptive regularization](https://dl.acm.org/citation.cfm?id=2124313)

参考[https://blog.csdn.net/google19890102/article/details/73301949](https://blog.csdn.net/google19890102/article/details/73301949)

这几篇paper的pdf可以从这里下载：[https://github.com/buptjz/Factorization-Machine](https://github.com/buptjz/Factorization-Machine)

可以发现，FM里的正则有一坨`\(\lambda\)`，

然后有一篇Incremental Factorization Machines for Persistently Cold-starting Online Item Recommendation

简单地理解，就是adaptive regularization是先更新参数，然后更新lambda；而这篇是借鉴了另一篇的，先看在测试集的效果，再更新参数的思想，所以一次处理一条，这条先拿来更新lambda，然后用更新完的lambda来更新参数。。

