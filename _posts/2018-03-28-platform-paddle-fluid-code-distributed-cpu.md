---
layout: post
category: "platform"
title: "paddle fluid分布式cpu相关"
tags: [paddle, fluid, 分布式, cpu, fleet, ]
---

目录

<!-- TOC -->

- [简介](#简介)

<!-- /TOC -->

## 简介

[ERNIE2.0背后的神助攻：飞桨高性能分布式训练引擎](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650767419&idx=2&sn=178f83aaae05eb31b7edfbc78be0be3a&chksm=871a4445b06dcd53f58476f35bcada44c7e7f7d874374e1c8b78e3ee14d93308d08cfccacd2b&scene=0&xtrack=1&pass_ticket=Kz97uXi0CH4ceADUC3ocCNkjZjy%2B0DTtVYOM7n%2FmWttTt5YKTC2DQT9lqCel7dDR#rd)

+ ```./paddle/fluid/framework/dist_multi_trainer.cc```：入口
+ ```./paddle/fluid/framework/downpour_worker.cc```：pull push sparse
+ ```./paddle/fluid/framework/pull_dense_worker.cc```：pull，push dense
+ ```./paddle/fluid/framework/data_feed.cc```：解析ins
