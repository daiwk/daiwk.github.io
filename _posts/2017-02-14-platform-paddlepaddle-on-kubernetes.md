---
layout: post
category: "platform"
title: "paddlepaddle on kubernetes"
tags: [paddlepaddle, kubernetes]
---

目录

<!-- TOC -->

- [paddle+k8s](#paddlek8s)

<!-- /TOC -->

paddlepaddle和kubernetes结合：
[http://blog.kubernetes.io/2017/02/run-deep-learning-with-paddlepaddle-on-kubernetes.html](http://blog.kubernetes.io/2017/02/run-deep-learning-with-paddlepaddle-on-kubernetes.html)

## paddle+k8s

1期：
功能点：
内嵌式服务发现：能获取所有endpoints
参数服务器、master、trainer可以自动部署

2期：
基于paddle的notebook的学习体验的过程，然后可以使用最新的client，启动大规模的容器。
数据放在bos上

参考[https://github.com/PaddlePaddle/Paddle/blob/develop/doc/howto/usage/k8s/k8s_distributed_cn.md](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/howto/usage/k8s/k8s_distributed_cn.md)

caffe->paddle:
[https://github.com/PaddlePaddle/models/tree/develop/image_classification/caffe2paddle](https://github.com/PaddlePaddle/models/tree/develop/image_classification/caffe2paddle)


