---
layout: post
category: "nlp"
title: "新型机器翻译"
tags: [MUSE, facebook, gan, 机器翻译, 无监督]
---

目录

<!-- TOC -->

- [基于GAN的机器翻译](#基于gan的机器翻译)
    - [seqgan](#seqgan)
    - [Adversarial-NMT](#adversarial-nmt)
- [无监督机器翻译](#无监督机器翻译)
    - [MUSE](#muse)
    - [基于短语和神经的无监督机器翻译](#基于短语和神经的无监督机器翻译)

<!-- /TOC -->

## 基于GAN的机器翻译

汇总[https://zhuanlan.zhihu.com/p/30788930](https://zhuanlan.zhihu.com/p/30788930)

### seqgan

[SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)

### Adversarial-NMT

参考知乎文章：[https://zhuanlan.zhihu.com/p/26661225](https://zhuanlan.zhihu.com/p/26661225)

[Adversarial Neural Machine Translation](https://arxiv.org/pdf/1704.06933.pdf)


## 无监督机器翻译

### MUSE

[Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087)

[Unsupervised Machine Translation With Monolingual Data Only](https://arxiv.org/abs/1711.00043)

[https://github.com/facebookresearch/MUSE](https://github.com/facebookresearch/MUSE)

### 基于短语和神经的无监督机器翻译

[学界 \| FAIR新一代无监督机器翻译：模型更简洁，性能更优](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650741621&idx=4&sn=210c9b3694f56508267e95235fb99f3c&chksm=871adf0bb06d561dc99a89a5f56cf835bc92aa25401294216262128402490918d2787fe7fd09&mpshare=1&scene=1&srcid=0429xcZSkAituLbHIsmgNBAb&pass_ticket=INCrGaryVZRn7Xp0qFQ7uod1VN14o8mkpvq1bswtroEgKQavvDm7mmg4E7yTOH6d#rd)

[Phrase-Based & Neural Unsupervised Machine Translation](https://arxiv.org/pdf/1804.07755.pdf)

机器翻译系统在某些语言上取得了接近人类水平的性能，但其有效性在很大程度上依赖**大量双语文本**，这降低了机器翻译系统在大多数语言对中的适用性。本研究探讨了如何在**只有大型单语语料库**的情况下进行机器翻译。

在广泛使用的 WMT'14 英法和 WMT'16 德英基准测试中，我们的模型在不使用平行语句的情况下分别获得 27.1 和 23.6 的 BLEU 值，比当前最优技术高 11 个 BLEU 点。

大量文献研究了在**有限的监督**下使用**单语数据**来提升翻译性能的问题。这种有限的监督通常是以下形式：

+ 一小组平行句子（Sennrich et al., 2015a; Gulcehre et al., 2015; He et al., 2016; Gu et al., 2018; Wang et al., 2018）
+ 使用其他相关语言的一大组平行句子（Firat et al., 2016; Johnson et al., 2016; Chen et al., 2017; Zheng et al., 2017）
+ 双语词典（Klementiev et al., 2012; Irvine and Callison-Burch, 2014, 2016）
+ 可比语料库（Munteanu et al., 2004; Irvine and Callison-Burch, 2013）

最近，研究者提出了两种**完全无监督**的方法（Lample et al., 2018; Artetxe et al., 2018），**仅依赖于每种语言**的**单语语料库**，如 Ravi 和 Knight（2011）的开创性研究。

虽然这两项研究存在细微的技术差异，但我们发现了它们成功的几个共同因素:

+ 它们使用**推断的双语词典**仔细完成模型的**初始化**。
+ 它们利用强大的语言模型，通过训练**序列到序列**的系统（Sutskever et al., 2014; Bahdanau et al., 2015）作为**去噪自编码器**（Vincent et al., 2008）
+ 通过**回译**自动**生成句对**，将无监督问题**转化为监督问题**（Sennrich et al., 2015a）回译过程的关键是维护**两个模型**，一个**将源语翻译成目标语言**，另一个则是**从目标语言生成源语言**。前者生成数据，用于训练后者，反之亦然。
+ 这些模型限制编码器产生的、在两种语言之间共享的潜在表征。将这些片段放在一起，无论输入语言是什么，编码器都会产生类似的表征。解码器既作为语言模型又作为噪声输入的翻译器进行训练，它学习与后向模型（从目标语到源语的操作）一起产生越来越好的译文。这种**迭代过程**在完全无监督的环境中取得了良好的结果，例如，它在 WMT'14 英法基准测试中的 BLEU 值达到了～15。



