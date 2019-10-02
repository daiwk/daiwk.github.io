---
layout: post
category: "dl"
title: "深度学习加速tricks"
tags: [加速, speed up, ]
---

目录

<!-- TOC -->


<!-- /TOC -->


参考[26秒单GPU训练CIFAR10，Jeff Dean也点赞的深度学习优化技巧](https://mp.weixin.qq.com/s/Mv7QTTGo3WIsQFmXN4fD5g)

colab地址：[https://colab.research.google.com/github/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb](https://colab.research.google.com/github/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb)

blog地址：[https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/](https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/)

主要通过以下几步将时间减小到26s：

+ GPU 上进行数据预处理 (70s)
+ 更换最大池化顺序 (64s)
+ 标签平滑化 (59s)
+ 使用 CELU 激活函数 (52s)
+ 幽灵批归一化 (46s)
+ 固定批归一化的缩放 (43s)
+ 输入 patch 白化 (36s)
+ 指数移动平均时间 (34s)
+ 测试状态增强 (26s)
