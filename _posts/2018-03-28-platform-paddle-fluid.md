---
layout: post
category: "platform"
title: "paddle fluid"
tags: [paddle, fluid]
---

目录

<!-- TOC -->

- [简介](#%E7%AE%80%E4%BB%8B)
- [核心概念](#%E6%A0%B8%E5%BF%83%E6%A6%82%E5%BF%B5)

<!-- /TOC -->


## 简介

参考[https://daiwk.github.io/posts/platform-tensorflow-folding.html](https://daiwk.github.io/posts/platform-tensorflow-folding.html)

先了解下eager execution的优点：

+ 快速调试即刻的运行错误并通过 Python 工具进行整合
+ 借助易于使用的 Python 控制流支持动态模型
+ 为自定义和高阶梯度提供强大支持
+ 适用于几乎所有可用的 TensorFlow 运算

fluid也有点类似，分为编译时和运行时。

编译时：

+ 创建变量描述Variable
+ 创建operators的描述OpDesc
+ 创建operator的属性
+ 推断变量的类型和形状，进行静态检查：InferShape
+ 规划变量的内存复用
+ 创建反向计算
+ 添加优化相关的Operators
+ (可选)添加多卡/多机相关的Operator，生成在多卡/多机上运行的程序


运行时：
+ 创建Executor
+ 为将要执行的一段计算，在层级式的Scope空间中创建Scope
+ 创建Block，依次执行Block

<html>
<br/>

<img src='../assets/fluid-compile-run.png' style='max-height: 250px'/>
<br/>

</html>

另外，fluid自己封装了各种switch/ifelse/while_op等。

## 核心概念



