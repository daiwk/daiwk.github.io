---
layout: post
category: "platform"
title: "horovod"
tags: [horovod, ]
---

目录

<!-- TOC -->

- [安装](#%e5%ae%89%e8%a3%85)
  - [有一个装了tf的py](#%e6%9c%89%e4%b8%80%e4%b8%aa%e8%a3%85%e4%ba%86tf%e7%9a%84py)
  - [安装openmpi](#%e5%ae%89%e8%a3%85openmpi)
  - [pip安装](#pip%e5%ae%89%e8%a3%85)

<!-- /TOC -->

[https://juejin.im/post/5cbc6dbd5188253236619ccb#heading-9](https://juejin.im/post/5cbc6dbd5188253236619ccb#heading-9)

参考[[深度学习] 分布式Horovod介绍（四）](https://blog.csdn.net/zwqjoy/article/details/89552432)

主页：

[https://github.com/uber/horovod](https://github.com/uber/horovod)

## 安装

### 有一个装了tf的py

例如：

```
/home/work/tools/python-2.7.8-tf1.4-gpu
```

### 安装openmpi

下载地址：
[https://www.open-mpi.org/software/ompi/v3.0/](https://www.open-mpi.org/software/ompi/v3.0/)

解压后
```
./configure --prefix=/home/work/tools/openmpi
make && make install
```

### pip安装

确保PATH里有gcc48，以及```/home/work/tools/openmpi/bin/```

如果是GPU的，确保```export LD_LIBRARY_PATH=/home/work/cudnnv6/cuda/lib64/:$LD_LIBRARY_PATH```

因为安装时要用到```-lpython2.7```，如果没有root权限，去报错的gcc命令里找```-L```的路径，发现最简单粗暴的方法就是

```
cp  /home/work/tools/python-2.7.8-tf1.4-gpu/lib/libpython2.7.so* /home/work/tools/openmpi/lib
```

或者

```
cp /home/work/tools/python-2.7.8-tf1.4-gpu/lib/ /home/work/tools/python-2.7.8-tf1.4-gpu/lib/python2.7/site-packages/tensorflow
```

然后pip安装：

```
/home/work/tools/python-2.7.8-tf1.4-gpu/bin/python  /home/work/tools/python-2.7.8-tf1.4-gpu/bin/pip install horovod
```
