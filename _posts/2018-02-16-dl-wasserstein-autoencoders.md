---
layout: post
category: "dl"
title: "Wasserstein自编码器"
tags: [Wasserstein自编码器, ]
---

目录

<!-- TOC -->


<!-- /TOC -->


[ICLR 2018 \| 谷歌大脑Wasserstein自编码器：新一代生成模型算法](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650737956&idx=3&sn=d3597e73bd7457d608488cf1087389f4&chksm=871ac95ab06d404c1952b6fd59d0ec59267e5647cd52b39d2a9e8adf070b2c4c372dd8f0e219&scene=0&pass_ticket=52szIHU2nDe0%2FHpNVg6A0uKH8OcCUpRBXt0cv3flp56HR4%2FKKYQVtbTX3H73ePQ7#rd)

变分自编码器（VAE）与生成对抗网络（GAN）是复杂分布上无监督学习主流的两类方法。近日，谷歌大脑 Ilya Tolstikhin 等人提出了又一种新思路：Wasserstein 自编码器，其不仅具有 VAE 的一些优点，更结合了 GAN 结构的特性，可以实现更好的性能。该研究的论文《Wasserstein Auto-Encoders》已被即将在 4 月 30 日于温哥华举行的 ICLR 2018 大会接收。