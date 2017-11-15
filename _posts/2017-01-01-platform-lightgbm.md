---
layout: post
category: "platform"
title: "lightgbm"
tags: [lightgbm, ]
---


目录

<!-- TOC -->

- [安装](#安装)
    - [安装cxx包](#安装cxx包)
    - [安装py](#安装py)
    - [验证](#验证)

<!-- /TOC -->
[https://github.com/Microsoft/LightGBM](https://github.com/Microsoft/LightGBM)

参考[http://www.jianshu.com/p/48e82dbb142b](http://www.jianshu.com/p/48e82dbb142b)

## 安装

### 安装cxx包

```
apt-get update
apt-get install git
apt-get install cmake
apt-get install openmpi-bin openmpi-doc libopenmpi-dev
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build
cmake -DUSE_MPI=ON ..
make -j4
```

### 安装py

```
pip install lightgbm
```

### 验证

```
import lightgbm as lgb
```

