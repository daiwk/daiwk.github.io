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
    - [nnlm综述](#nnlm综述)
    - [阅读理解综述](#阅读理解综述)
    - [ACL2019预训练语言模型](#acl2019预训练语言模型)

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

### nnlm综述

[从经典结构到改进方法，神经网络语言模型综述](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650766368&idx=4&sn=d7876bb4adb0ded6ad3736b878f8e541&chksm=871ab85eb06d31481e25aea171f87d355ac25465dc173abe49a1e075463cb19b777d1776b4dd&scene=0&xtrack=1&pass_ticket=I7vMVoY36Vu5%2FFz%2FMUDKXgy%2FHocjPiCFYYtVANqq1m0CCQBpIAQhSU5BGMcu7Il0#rd)

### 阅读理解综述

[神经机器阅读理解最新综述：方法和趋势](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247498503&idx=1&sn=ca27b9f04effcdfd8add3cd22aede262&chksm=96ea2487a19dad91aa546eb1d49d851d3e43b56f360ad75bd7bb2d18a2977c16b1908ce1b43f&scene=0&xtrack=1&pass_ticket=I7vMVoY36Vu5%2FFz%2FMUDKXgy%2FHocjPiCFYYtVANqq1m0CCQBpIAQhSU5BGMcu7Il0#rd)

### ACL2019预训练语言模型

[ACL 2019提前看：预训练语言模型的最新探索](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650766791&idx=4&sn=c2e2088877e0ec88fe67e68ed7db5f65&chksm=871ab9b9b06d30afaeb21aefff562d6d3eab86f82b0b0a86315e6c75f54171e2d6ddaf2d0199&scene=0&xtrack=1&pass_ticket=I7vMVoY36Vu5%2FFz%2FMUDKXgy%2FHocjPiCFYYtVANqq1m0CCQBpIAQhSU5BGMcu7Il0#rd)
