---
layout: post
category: "platform"
title: "tf step by step: chap 2. 对比"
tags: [tf step by step,]
---

目录

<!-- TOC -->

- [1. tensorflow](#1-tensorflow)

<!-- /TOC -->

-- 整理自《tensorflow实战》 chap 2

## 1. tensorflow

和Theano一样支持自动求导，用户不用通过bp求梯度。和Caffe一样，核心代码是用C++写的，简化上线部署的复杂度（手机这种内存&cpu都紧张的嵌入式设备，也可以直接用C++接口运行复杂模型）。tf还通过swig，提供了官方的py/go/java接口。

使用python时，有一个影响效率的问题：每个mini-batch要从python中feed到网络中，
