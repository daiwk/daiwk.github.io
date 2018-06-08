---
layout: post
category: "knowledge"
title: "gperftools"
tags: [gperftools, ]
---

目录

<!-- TOC -->

- [内存泄漏](#内存泄漏)
- [常用工具](#常用工具)
    - [gperftools的功能支持](#gperftools的功能支持)

<!-- /TOC -->

参考[高阶干货\｜如何用gperftools分析深度学习框架的内存泄漏问题](https://mp.weixin.qq.com/s?__biz=MzIxNTgyMDMwMw==&mid=2247484403&idx=1&sn=5b260e7d681a4550811ee5611a1dd4ce&chksm=97933293a0e4bb85104066213606f5c89002fe78fa0518d95c71918a5a6ae656c62beac0f1ae&mpshare=1&scene=1&srcid=0608zWg71wg9411XBaYQ7o1V&pass_ticket=xLsJxSJh9Kgj4HKrq0S6VH1cKTCnSBShWGuwGJy9Gfpbp1CgoA6crqJiPhq9JjnM#rd)

### 内存泄漏

内存泄漏一般是由于程序**在堆(heap)上**分配了内存而没有释放，随着程序的运行占用的内存越来越大，一方面会影响程序的稳定性，可能让运行速度越来越慢，或者造成oom，甚至会影响程序所运行的机器的稳定性，造成宕机。

### 常用工具

+ valgrind直接分析非常困难，需要自己编译debug版本的、带valgrind支持的专用Python版本，而且输出的信息中大部分是Python自己的符号和调用信息，很难看出有用的信息，另外**使用valgrind会让程序运行速度变得非常慢**，所以不建议使用。
+ gperftools使用简单，无需重新编译代码即可运行，对运行速度的影响也比较小。

#### gperftools的功能支持

gperftool主要支持以下四个功能：

+ thread-caching malloc
+ heap-checking using tcmalloc
+ heap-profiling using tcmalloc
+ CPU profiler

