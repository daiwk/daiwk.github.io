---
layout: post
category: "knowledge"
title: "cuda"
tags: [cuda, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

首先，下载 cuda 的 driver，例如440.33.01版本的 driver：

```shell
https://www.nvidia.com/download/driverResults.aspx/154570
```

然后用root安装：

```shell
sh NVIDIA-Linux-x86_64-440.33.01.run --no-kernel-module 
```

然后应该就可以```nvidia-smi```了，只是出来的可能是：

```shell
NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: N/A 
```

然后，安装 cuda，例如440.33.01版本对应的 cuda10.2：

```shell
https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=6&target_type=runfilelocal
```

然后用root：

```shell
sh cuda_10.2.89_440.33.01_rhel6.run -tmpdir /home/work/tmp -s --toolkit  
```

再然后：

```shell
[work@xxx.xxx ~]$  nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:24:38_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
```
