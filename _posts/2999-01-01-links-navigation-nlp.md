---
layout: post
category: "links"
title: "【置顶】导引——nlp论文集合"
tags: [nlp, ]
---

目录

<!-- TOC -->

- [机器翻译](#机器翻译)
    - [10篇必读](#10篇必读)
- [其他](#其他)

<!-- /TOC -->

## 机器翻译

参考清华机器翻译组整理的[https://github.com/THUNLP-MT/MT-Reading-List](https://github.com/THUNLP-MT/MT-Reading-List)

### 10篇必读

+ [The Mathematics of Statistical Machine Translation: Parameter Estimation](http://aclweb.org/anthology/J93-2003)
+ [BLEU: a Method for Automatic Evaluation of Machine Translation](http://aclweb.org/anthology/P02-1040)
+ [Statistical Phrase-Based Translation](http://aclweb.org/anthology/N03-1017)
+ [Minimum Error Rate Training in Statistical Machine Translation](http://aclweb.org/anthology/P03-1021)
+ [Hierarchical Phrase-Based Translation](http://aclweb.org/anthology/J07-2003)
+ [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
+ [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
+ [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980)
+ [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)
+ [Attention is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)


## 其他

[A Tensorized Transformer for Language Modeling](https://arxiv.org/pdf/1906.09777.pdf)

多头注意力机制限制了模型的发展，使得模型需要较大的算力支持。为了解决这一问题，基于**张量分解**和**参数共享**的思想，本文提出了**多头线性注意力（Multi-linear attention）**和**Block-Term Tensor Decomposition（BTD）**。研究人员在语言建模任务及神经翻译任务上进行了测试，与许多语言建模方法相比，多头线性注意力机制不仅可以**大大压缩模型参数数量**，而且**提升了模型的性能**。

分解那块参考的[Tensor decompositions and applications](http://www.kolda.net/publication/TensorReview.pdf)和[Decompositions of a higher-order tensor in block terms—part ii: Definitions and uniqueness](https://www.researchgate.net/publication/220656664_Decompositions_of_a_Higher-Order_Tensor_in_Block_Terms-Part_II_Definitions_and_Uniqueness?_iepl%5BgeneralViewId%5D=GidOAKvMwvySBJxXGLf7dnk20JGueu7IopW8&_iepl%5Bcontexts%5D%5B0%5D=searchReact&_iepl%5BviewId%5D=7E2q1w0hmoIJSdCgfRKWh6JguNUiAGrpN0Lk&_iepl%5BsearchType%5D=publication&_iepl%5Bdata%5D%5BcountLessEqual20%5D=1&_iepl%5Bdata%5D%5BinteractedWithPosition1%5D=1&_iepl%5Bdata%5D%5BwithoutEnrichment%5D=1&_iepl%5Bposition%5D=1&_iepl%5BrgKey%5D=PB%3A220656664&_iepl%5BtargetEntityId%5D=PB%3A220656664&_iepl%5BinteractionType%5D=publicationTitle)，后者好像没有pdf可以下载。。

