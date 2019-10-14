---
layout: post
category: "dl"
title: "Arror数据集+tf"
tags: [arrow, tensorflow, tf, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

[TensorFlow 与 Apache Arrow 数据集搭配简介](https://mp.weixin.qq.com/s/O2mfxaczHQYHeUsdf4Ul1w)

Apache Arrow 本质上是一种基于内存的列式数据的标准格式，旨在提高系统之间的效率和互操作性。

+ 首先，作为一种数据标准，无论数据来源为何，Apache Arrow 均能确保数据类型安全性和数据完整性。
+ 其次，作为一种内存格式，Apache Arrow 允许系统之间进行数据交换，而无需为不同文件格式进行序列化或转换。
+ 最后，Arrow 始终针对数据处理进行全面优化，无论是零拷贝中读取还是现代硬件上的加速操作支持均涵盖在内。

因此，这确保了您可以高效地处理数据，同时与不同规模的各类系统无缝集成。

Arrow 数据集是 tf.data.Dataset 的扩展，因此两者可利用相同的 API 与 tf.data 流水线集成，并可作为 tf.keras 的输入。TensorFlow I/O 目前提供 3 种 Arrow 数据集，按名称排序如下：ArrowDataset、ArrowFeatherDataset 和 ArrowStreamDataset。这三种数据集均由相同的底层 Arrow 数据馈送，且此类底层数据具有两个重要特征：结构化 和 批量化。


[TensorFlow 与 Apache Arrow 数据集搭配最佳实践](https://mp.weixin.qq.com/s/5UtpFg7Zmm6WY0OOxzeueQ)

