---
layout: post
category: "nlp"
title: "Deep Recurrent Models with Fast-Forward Connections for Neural Machine Translation(rnn_nmt_baidu)"
tags: [nlp, natural language processing, nmt, lstm nmt, 机器翻译]
---

这篇论文发表在acl,2016上
[论文地址](../assets/Deep Recurrent Models with Fast-Forward Connections for Neural Machine Translation.pdf)

# **0. 摘要**

基于**deep lstm networks + interleaved(插入/交错) deep bi-lstm**，使用了新的linear connections**(fast-forward connections)**.fast-forward connections在**propagating gradient**以及建立**深度达到16的深度拓扑**中起到了重要作用。

在wmt'14的English->French的翻译中，**单一attention**的模型BLEU达到37.7（超越了传统nmt的单浅层模型6.2的BLEU）；去掉attention，BLEU=36.3。在**对unknown words进行了特殊的处理**，同时进行**模型ensemble**之后，可以达到BLEU=40.4。


# **1. 介绍**

传统mt模型（statistical mt，SMT）包括了multiple separately tuned components，而NMT将源序列encode到一个**continuous representation space**，然后使用end-to-end的方式生成新的序列。

NMT一般有两种拓扑：encoder-decoder network([Sutskever et al., 2014](../assets/sequence to sequence learning with neural networks.pdf))以及attention网络（[Bahdanau et al., 2015](../assets/neural machine translation by jointly learning to align and translate.pdf)）。

encoder-decoder网络将源序列表示成一个**fixed dimensional vector**，并**word by word**地生成目标序列。

attention网络使用**all time steps的输入**建立一个**targetwords和inputwords之间**的detailed relationship。

但single的neural network和最好的conventional(传统) SMT还是不能比的，6层BLEU才只有31.5，但传统方法有37.0。

