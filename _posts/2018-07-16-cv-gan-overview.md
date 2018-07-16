---
layout: post
category: "cv"
title: "gan landscape"
tags: [gan landscape, ]
---

目录

<!-- TOC -->

- [1. 概述](#1-%E6%A6%82%E8%BF%B0)
- [x. 代码](#x-%E4%BB%A3%E7%A0%81)

<!-- /TOC -->


## 1. 概述

[谷歌大脑发布GAN全景图：看百家争鸣的生成对抗网络](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650745273&idx=1&sn=b057305f7431423adebaa519dcd23547&chksm=871aedc7b06d64d1f66d78687a721b7360f0b11f20b44b00c7d92047f137e75be20d9ffb2009&mpshare=1&scene=1&srcid=0715c3IDUOvpvCyTHgZhMDMV&pass_ticket=qgYs5vOKlc87Cj4B5uTln9ELDfWQJnTqwJO%2B5ipNoI6K7VStQ9djW9PXdfzSwMD3#rd)

有以下两篇：

+ [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337)
+ [The GAN Landscape: Losses, Architectures, Regularization, and Normalization](https://arxiv.org/abs/1807.04720)


训练 GAN 需要在生成器和判别器的参数上求解一个**极小极大问题**。由于生成器和判别器通常被参数化为深度卷积神经网络，这个极小极大问题在实践中非常困难。

作者主要从损失函数、判别器的正则化与归一化、生成器与判别器的架构、评估度量与数据集等 5 个方面进行了讨论。

+ **损失函数：**作者讨论了原版 GAN 的 JS 距离、WGAN 的 Wasserstein 距离和最小二乘等损失函数。
+ **判别器的正则化与归一化**：
  + **判别器的正则化**主要为梯度范数罚项，例如在 WGAN 中，这种梯度范数惩罚主要体现在对违反 1-Lipschitzness 平滑的软惩罚。此外，模型还能根据数据流形评估梯度范数惩罚，并鼓励判别器在该数据流形上成分段线性。
  + **判别器的归一化**主要体现在最优化与表征上，即归一化能获得更高效的梯度流与更稳点的优化过程，以及修正各权重矩阵的谱结构而获得更更丰富的层级特征。
+ **生成器与判别器架构**：
  + **深度卷积生成对抗网络**：生成器与判别器分别包含 5 个卷积层，且带有谱归一化的变体称为 SNDCGAN。
  + **残差网络**：ResNet19 的生成器包含 5 个残差模块，判别器包含 6 个残差模块。
+ **评估度量**：包括 Inception Score (IS)、Frechet Inception Distance (FID) 和 Kernel Inception distance (KID) 等，它们都提供生成样本质量的定量分析
+ **数据集**：IFAR10、CELEBA-HQ-128 和 LSUN-BEDROOM。


## x. 代码

代码：[https://github.com/google/compare_gan](https://github.com/google/compare_gan)
安装： clone下来

然后需要修改一下setup.py，改为：

```shell
    scripts=[
        'compare_gan/bin/compare_gan_generate_tasks',
        'compare_gan/bin/compare_gan_prepare_datasets.sh',
        'compare_gan/bin/compare_gan_run_one_task',
        'compare_gan/bin/compare_gan_run_test.sh',
    ],
```

然后安装

```shell
pip install -e .
```

然后运行下面的代码，把数据集准备好

```shell
cd bin && bash -x compare_gan_prepare_datasets.sh
## 可能需要修改一下t2t_datagen的路径，例如：
#T2T_DATAGEN="$HOME/.local/bin/t2t-datagen"
#T2T_DATAGEN="/usr/local/lib/python3.6/site-packages/tensor2tensor/bin/t2t_datagen.py"
```

注意，这两个数据集没装：

+ Lsun bedrooms dataset: If you want to install lsun-bedrooms you need to run t2t-datagen yourself (this dataset will take couple hours to download and unpack).

+ CelebaHQ dataset: currently it is not available in tensor2tensor. Please use the ProgressiveGAN [https://github.com/tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans) for instructions on how to prepare it.

然后就可以跑了(```compare_gan_generate_tasks```和```compare_gan_run_one_task```是安装的两个bin)

```shell
# Create tasks for experiment "test" in directory /tmp/results. See "src/generate_tasks_lib.py" to see other possible experiments.
compare_gan_generate_tasks --workdir=/tmp/results --experiment=test

# Run task 0 (training and eval)
compare_gan_run_one_task --workdir=/tmp/results --task_num=0 --dataset_root=/tmp/datasets

# Run task 1 (training and eval)
compare_gan_run_one_task --workdir=/tmp/results --task_num=1 --dataset_root=/tmp/datasets
```
