---
layout: post
category: "dl"
title: "分布式深度学习"
tags: [分布式深度学习, ]
---

目录

<!-- TOC -->

- [DC-ASGD算法：补偿异步通信中梯度的延迟](#dc-asgd算法补偿异步通信中梯度的延迟)
- [Ensemble-Compression算法：改进非凸模型的聚合方法](#ensemble-compression算法改进非凸模型的聚合方法)
- [随机重排下算法的收敛性分析：改进分布式深度学习理论](#随机重排下算法的收敛性分析改进分布式深度学习理论)

<!-- /TOC -->

参考[分布式深度学习新进展：让“分布式”和“深度学习”真正深度融合](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652021689&idx=4&sn=a32b3a89ec9b27fb3ae0f14636aebac5&chksm=f121d748c6565e5ea1c22496a8b8b58029b50b23e7ba4f44fd15e98fb4225f129506b1b681c3&mpshare=1&scene=1&srcid=07086KpHqOtQ4edEpGgxvRyJ&pass_ticket=MbycpNZtuIW06kKTrmdAWLocG4d06ASGMcDZkz7VwgG9rlL9Wj9iLFK58BCCqJP6#rd)

## DC-ASGD算法：补偿异步通信中梯度的延迟

**Asynchronous Stochastic Gradient Descent with Delay Compensation, ICML2017**

其实这个在tensorRS里提到了呢

[https://daiwk.github.io/posts/platform-tensorflow-optimizations.html#%E6%A2%AF%E5%BA%A6%E8%A1%A5%E5%81%BF](https://daiwk.github.io/posts/platform-tensorflow-optimizations.html#%E6%A2%AF%E5%BA%A6%E8%A1%A5%E5%81%BF)



## Ensemble-Compression算法：改进非凸模型的聚合方法

**Ensemble-Compression: A New Method for Parallel Training of Deep Neural Networks, ECML 2017**


## 随机重排下算法的收敛性分析：改进分布式深度学习理论

**Convergence Analysis of Distributed Stochastic Gradient Descent with Shuffling**

[https://arxiv.org/abs/1709.10432](https://arxiv.org/abs/1709.10432)
