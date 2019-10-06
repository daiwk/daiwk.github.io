---
layout: post
category: "links"
title: "【置顶】导引——nlp论文集合"
tags: [nlp, ]
---

目录

<!-- TOC -->

- [机器翻译](#%e6%9c%ba%e5%99%a8%e7%bf%bb%e8%af%91)
  - [10篇必读](#10%e7%af%87%e5%bf%85%e8%af%bb)
- [其他](#%e5%85%b6%e4%bb%96)
  - [nnlm综述](#nnlm%e7%bb%bc%e8%bf%b0)
  - [阅读理解综述](#%e9%98%85%e8%af%bb%e7%90%86%e8%a7%a3%e7%bb%bc%e8%bf%b0)
  - [ACL2019预训练语言模型](#acl2019%e9%a2%84%e8%ae%ad%e7%bb%83%e8%af%ad%e8%a8%80%e6%a8%a1%e5%9e%8b)
  - [nlp综述](#nlp%e7%bb%bc%e8%bf%b0)
  - [nlp路线图](#nlp%e8%b7%af%e7%ba%bf%e5%9b%be)
    - [Probability & Statistics](#probability--statistics)
    - [Machine Learning](#machine-learning)
    - [Text Mining](#text-mining)
    - [Natural Language Processing](#natural-language-processing)
  - [teacher forcing](#teacher-forcing)

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

### nlp综述

[A Survey of the Usages of Deep Learning in Natural Language Processing](https://arxiv.org/pdf/1807.10854v2)

过去数年，深度学习模型的爆炸式使用推动了自然语言处理领域的发展。在本文中，研究者简要介绍了自然语言处理领域的基本情况，并概述了深度学习架构和方法。然后，他们对近来的研究进行了筛选，对大量相关的文献展开总结。除了诸多计算机语言学的应用之外，研究者还分析研究了语言处理方面的几个核心问题。最后他们讨论了当前 SOTA 技术和方法并对该领域未来的研究提出了建议。

### nlp路线图

[https://github.com/graykode/nlp-roadmap](https://github.com/graykode/nlp-roadmap)

#### Probability & Statistics

<html>
<br/>
<img src='../assets/prob-stats.png' style='max-height: 400px'/>
<br/>
</html>

#### Machine Learning

<html>
<br/>
<img src='../assets/ml.png' style='max-height: 400px'/>
<br/>
</html>

#### Text Mining

<html>
<br/>
<img src='../assets/textmining.png' style='max-height: 400px'/>
<br/>
</html>

#### Natural Language Processing

<html>
<br/>
<img src='../assets/nlp.png' style='max-height: 400px'/>
<br/>
</html>

### teacher forcing

参考[ACL2019最佳论文冯洋：Teacher Forcing亟待解决 ，通用预训练模型并非万能](https://mp.weixin.qq.com/s/RNaFuacpbQ32OtIUthcPRQ)

[Bridging the Gap between Training and Inference for Neural Machine Translation](https://arxiv.org/abs/1906.02448)

提出了使用 Oracle 词语，用于替代 Ground Truth 中的词语，作为训练阶段约束模型的数据。

选择 Oracle Word 的方法有两种，一种是选择 word-level oracle，另一种则是 sentence-level oracle。

word-level oracle 的选择方法，在时间步为 j 时，获取前一个时间步模型预测出的每个词语的预测分数。为了提高模型的鲁棒性，论文在预测分数基础上加上了 Gumbel noise，最终取分数最高的词语作为此时的 Oracle Word。

sentence-level oracle 的选择方法则是在训练时，在解码句子的阶段，使用集束搜索的方法，选择集束宽为 k 的句子（即 top k 个备选句子），然后计算每个句子的 BLEU 分数，最终选择分数最高的句子。

当然，这会带来一个问题，即每个时间步都需要获得该时间步长度上的备选句子，因此集束搜索获得的句子长度必须和时间步保持一致。如果集束搜索生成的实际句子超出或短于这一长度该怎么办？这里研究人员使用了「Force Decoding」的方法进行干预。而最终选择的 Oracle Word 也会和 Ground Truth 中的词语混合，然后使用衰减式采样（Decay Sampling）的方法从中挑选出作为约束模型训练的词。

机器之心：我们知道，这篇论文的基本思想是：不仅使用 Ground Truth 进行约束，在训练过程中，也利用训练模型预测出的上一个词语作为其中的备选词语，这样的灵感是从哪里得到的呢？

冯洋：我们很早就发现了这样一个问题——训练和测试的时候模型的输入是不一样的。我们希望模型在训练过程中也能够用到预测出的词语。看到最近一些周边的工作，我们慢慢想到，将 Ground Truth 和模型自己预测出的词一起以 Sampling 的方式输入进模型。

机器之心：刚才您提到有一些周边的工作，能否请您谈谈有哪些相关的论文？

冯洋：这些周边的论文在 Related Work 中有写到，这些工作的基本思想都是一样的，都是希望将预测出的词语作为模型输入。比如说，根据 DAD（Data as Demonstrator）的方法。这一方法将预测出的词语和后一个词语组成的词语对（word-pair）以 bigram 的方式输入作为训练实例加入。另一种是 Scheduled Sampling 的方式，也是用 Sampling 的方式，把预测出的词语作为输入加入到模型训练中。

机器之心：论文使用了两种方法实现将预测词语作为训练的输入，一种是在 Word-level 选择 Oracle Word，另一种是在 Sentence-level 选择 Oracle Sentence，能否请您详细介绍下 Sentence-level 的方法？

冯洋：Sentence-level 的方法可以简单理解为进行了一次解码。我们从句子中取出前 k 个候选译文。这里的 k 我们选择了 3，即 Top3 的句子。然后在这些句子中再计算他们的 BLEU 分数，并选择分数最高的句子，作为 Oracle Sentence。

机器之心：我们知道，论文中，在选择 Oracle Sentence 的过程中会进行「Force Decoding」。需要强制保证生成的句子和原有的句子保持一致的长度。您认为这样的方法会带来什么样的问题？

冯洋：这是强制模型生成和 Ground Truth 长度一样的句子。这样模型可能会生成一些原本并不是模型想生成的结果，这可能会带来一些问题。但是对于 Teacher Forcing 来说这是必须的，因为 Teacher Forcing 本身要求每一个词都要对应。所以说，虽然看起来我们干预了句子的生成，但是在 Teacher Forcing 的场景下，这种干预不一定是坏的。

机器之心：为什么说这样的干预不一定是坏的？

冯洋：我们需要留意的是，Force Decoding 的方法是在训练阶段进行的，如果训练中这样做了，模型就会逐渐地适应这个过程。另一方面，Force Decoding 可以平衡一些极端的生成结果。比如说，当句子长度为 10，但模型只生成了仅有 2 个词的句子，或者是模型生成了有 20 个词的句子，所以说 Force Decoding 也可以平衡这样的极端情况。在 Teacher Forcing 的场景下，这是一种折中的方法，不能完全说这样的方法是不好的。

