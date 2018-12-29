---
layout: post
category: "knowledge"
title: "tf安装"
tags: [tf安装, ]
---

目录

<!-- TOC -->

- [装备工作](#装备工作)
    - [bazel](#bazel)
    - [jdk1.8](#jdk18)
- [源码安装](#源码安装)
    - [clone](#clone)
    - [configure](#configure)
    - [生成pip_package](#生成pip_package)
        - [仅cpu](#仅cpu)
        - [gpu](#gpu)
    - [生成whl](#生成whl)
    - [安装c++库](#安装c库)
        - [c++版本](#c版本)
        - [c版本](#c版本)

<!-- /TOC -->

## 装备工作

### bazel

### jdk1.8

## 源码安装

### clone

```shell
git clone https://github.com/tensorflow/tensorflow 
```

### configure

```shell
./configure
```

注意，这里可以配置默认python路径

### 生成pip_package

#### 仅cpu

```shell
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```

#### gpu

```shell
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

### 生成whl

```shell
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

这样，就在```/tmp/tensorflow_pkg```生成了```tensorflow-xxxx-py2-none-any.whl```

### 安装c++库

#### c++版本

```shell
cd tensorflow
bazel build :libtensorflow_cc.so
```

产出在```bazel-bin/tensorflow/libtensorflow_cc.so```

#### c版本

```shell
cd tensorflow
bazel build :libtensorflow.so
```

产出在```bazel-bin/tensorflow/libtensorflow.so```
