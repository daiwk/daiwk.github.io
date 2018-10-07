---
layout: post
category: "nlp"
title: "treegan"
tags: [treegan, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

参考[学界 \| TreeGAN：为序列生成任务构建有句法意识的GAN](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650749532&idx=3&sn=16cad5039f02255a6bf9bb7059eaad88&chksm=871afe22b06d77345b186159de9e8803749b3024df173ebafc23f41b681ec675be66f88a1544&mpshare=1&scene=1&srcid=1006oVlYpuXj2BLG934rPBEJ&pass_ticket=QxrSV3DS%2FiS9KVGB05ngctujF130h1%2BGnKxMMG4QijDKs3TyHWK%2Fx1yQCCmdpdq%2B#rd)

[TreeGAN: Syntax-Aware Sequence Generation with Generative Adversarial Networks](https://arxiv.org/abs/1808.07582)

生成对抗网络是由生成网络和判别网络组成的无监督学习框架。我们将它们称为生成器（G）和判别器（D）。D 学着去区分某个数据实例是来自真实世界还是人为合成的。G 试图通过生成高质量的合成实例来迷惑 D。在 GAN 框架中，D 和 G 被不断地轮流训练直到它们达到纳什均衡。训练好的 GAN 会得到一个能够产生看起来与真实数据十分相似的高质量数据实例的生成器。

受到其在图像生成和相关领域取得的巨大成功的启发，GAN[1] 最近已经被扩展到序列生成任务中 [2，3]。用于序列生成的 GAN 在现实世界中有许多重要的应用。例如，为了给一个数据库构建一个良好的查询优化器，研究人员可能希望生成大量高质量的合成 SQL 查询语句对优化器进行基准对比测试。不同于图像生成任务，大多数语言都有其固有的语法或句法。现有的用于序列生成的 GAN 模型 [2，3，7] 主要着眼于如图 1a 所示的句法无关（grammar-free）的环境。这些方法试图从数据中学习到复杂的底层句法和语法模式，这通常是非常具有挑战性的，需要大量的真实数据样本才能取得不错的性能。在许多形式语言中，语法规则或句法（例如，SQL 句法，Python PL 句法）是预定义好的。将这样的句法引入到 GAN 的训练中，应该会得出一个具有句法意识的更好的序列生成器，并且在训练阶段显著缩小搜索空间。有句法意识的现有序列生成模型 [4] 主要是通过极大似然估计（MLE）进行训练的，它们高度依赖于真实数据样本的质量和数量。一些研究 [2，5] 表明，对抗性训练可以进一步提高基于极大似然估计的序列生成性能。即使有句法意识的现有序列生成方法引入了语法信息，其生成结果也可能不是最好的。

