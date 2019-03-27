---
layout: post
category: "knowledge"
title: "pytorch安装"
tags: [pytorch安装, ]
---

目录

<!-- TOC -->

- [准备工作](#%E5%87%86%E5%A4%87%E5%B7%A5%E4%BD%9C)
- [使用conda](#%E4%BD%BF%E7%94%A8conda)
- [非conda](#%E9%9D%9Econda)
- [厂内安装](#%E5%8E%82%E5%86%85%E5%AE%89%E8%A3%85)

<!-- /TOC -->

参考[https://github.com/pytorch/pytorch#from-source](https://github.com/pytorch/pytorch#from-source)

## 准备工作

```shell
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
```

首先需要3.5以上的cmake：[https://cmake.org/download/](https://cmake.org/download/)，例如[https://cmake.org/files/v3.5/cmake-3.5.2.tar.gz](https://cmake.org/files/v3.5/cmake-3.5.2.tar.gz)

然后

```shell
./bootstrap --prefix=/home/work/xxx
make
make install
```

## 使用conda

首先需要

```shell
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
```

然后在linux上的话：

```shell
# Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda90 # or [magma-cuda80 | magma-cuda92 | magma-cuda100 ] depending on your cuda version
```

然后安装

```shell
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

## 非conda

```shell
../python-2.7.14-tf-2.0/bin/python ../python-2.7.14-tf-2.0/bin/pip install pyyaml mkl mkl-include setuptools cmake cffi typing
```

## 厂内安装

```shell
./xxx/bin/python ./xxx/bin/pip install torch
/opt/compiler/gcc-4.8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib ./xxx/bin/python -c "import torch"
```