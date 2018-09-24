---
layout: post
category: "platform"
title: "tensorflow代码解析——概览"
tags: [tensorflow代码, 代码解析, 概览 ]
---

目录

<!-- TOC -->

- [简介](#%E7%AE%80%E4%BB%8B)
    - [总体结构](#%E6%80%BB%E4%BD%93%E7%BB%93%E6%9E%84)
    - [代码结构](#%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%84)
        - [tensorflow/core](#tensorflowcore)
        - [tensorflow/stream_executor](#tensorflowstreamexecutor)
        - [tensorflow/contrib](#tensorflowcontrib)
        - [tensroflow/python](#tensroflowpython)
        - [third_party](#thirdparty)

<!-- /TOC -->

参考：

+ [A tour through the TensorFlow codebase](http://public.kevinrobinsonblog.com/docs/A%20tour%20through%20the%20TensorFlow%20codebase%20-%20v4.pdf)
+ [『深度长文』Tensorflow代码解析（一）](https://zhuanlan.zhihu.com/p/25646408)
+ [『深度长文』Tensorflow代码解析（二）](https://zhuanlan.zhihu.com/p/25927956)
+ [『深度长文』Tensorflow代码解析（三）](https://zhuanlan.zhihu.com/p/25929909)
+ [『深度长文』Tensorflow代码解析（四）](https://zhuanlan.zhihu.com/p/25932160)
+ [『深度长文』Tensorflow代码解析（五）](https://zhuanlan.zhihu.com/p/26031658)

## 简介

### 总体结构

从底向上分为**设备管理**和**通信**层、**数据操作**层、**图计算**层、**API接口**层、**应用**层。

+ 底层设备通信层负责网络通信和设备管理。
    + 设备管理可以实现TF**设备异构**的特性，支持CPU、GPU、Mobile等不同设备。
    + 网络通信依赖**gRPC**通信协议实现不同设备间的数据传输和更新。
+ 数据操作层是Tensor的**OpKernels**实现。这些OpKernels**以Tensor为处理对象**，依赖**网络通信和设备内存分配**，实现了各种Tensor操作或计算。Opkernels不仅包含MatMul等**计算操作**，还包含Queue等**非计算操作**
+ 图计算层（Graph），包含**本地计算流图**和**分布式计算流图**的实现。Graph模块包含Graph的**创建**、**编译**、**优化**和**执行**等部分，Graph中**每个节点都是OpKernels类型表示**。
+ API接口层。**Tensor C API**是对TF功能模块的接口封装，便于其他语言平台调用。
+ 应用层。不同编程语言在应用层通过API接口层调用TF核心功能实现相关实验和应用。

### 代码结构

```以2018.09.23的master为基准```:

#### tensorflow/core

```tensorflow/core```目录包含了TF核心模块代码：

+ ```public```: API接口头文件目录，用于外部接口调用的API定义，主要是```session.h```。
+ ```client```: API接口实现文件目录。（目前已经没有这个目录了…）
+ ```platform```: ``OS系统``相关```接口文件```，如file system, env等。
+ ```protobuf```: 均为.proto文件，用于数据传输时的结构序列化。（都是proto3的语法）
+ ```common_runtime```: 公共运行库，包含```session```, ```executor```, ```threadpool```, ```rendezvous```, ```memory管理```, ```设备分配算法```等。
+ ```distributed_runtime```: 分布式执行模块，如```rpc session```, ```rpc master```, ```rpc worker```, ```graph manager```。
+ ```framework```: 包含基础功能模块，如```log```, ```memory```, ```tensor```
+ ```graph```: **计算流图**相关操作，如```construct```, ```partition```, ```optimize```, ```execute```等
+ ```kernels```: 核心Op，如```matmul```, ```conv2d```, ```argmax```, ```batch_norm```等
+ ```lib```: 公共基础库，如```gif```、```gtl(google模板库)```、```hash```、```histogram```、```jpeg```、```png```、```wav```等。
+ ```ops```: **基本**ops运算(```xxx_ops.cc```)，ops**梯度**运算（```xxx_grad.cc```），**io相关的ops***（```io_ops.cc```），***控制流和数据流***操作（```control_flow_ops.cc```和```data_flow_ops.cc```）

#### tensorflow/stream_executor

Tensorflow/stream_executor目录是并行计算框架，由google stream executor团队开发。

#### tensorflow/contrib

tensorflow/contrib目录是contributor开发目录。

#### tensroflow/python

tensroflow/python目录是python API客户端脚本

#### third_party

+ eigen3：eigen矩阵运算库，tf基础ops调用
+ gpus: 封装了cuda/cudnn编程库

