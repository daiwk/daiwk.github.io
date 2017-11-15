---
layout: post
category: "nlp"
title: "Deep Recurrent Models with Fast-Forward Connections for Neural Machine Translation(rnn_nmt_baidu)"
tags: [nlp, natural language processing, nmt, lstm nmt, 机器翻译]
---

目录

<!-- TOC -->

- [**0. 摘要**](#0-摘要)
- [**1. 介绍**](#1-介绍)
- [**2. NMT**](#2-nmt)
- [**3. Deep Topology**](#3-deep-topology)
    - [**3.1 Network**](#31-network)
    - [**3.2 Train technique**](#32-train-technique)
    - [**3.3 Generation**](#33-generation)
- [**4. Experiments**](#4-experiments)
    - [**4.1 Data sets**](#41-data-sets)
    - [**4.2 Model settings**](#42-model-settings)
    - [**4.3 Optimization**](#43-optimization)
    - [**4.4 Results**](#44-results)
        - [**4.4.1 Single models**](#441-single-models)
        - [**4.4.2 Post processing**](#442-post-processing)
    - [**4.5 Analysis**](#45-analysis)
        - [**4.5.1 Length**](#451-length)
        - [**4.5.2 Unknown words**](#452-unknown-words)
        - [**4.5.3 Over-fitting**](#453-over-fitting)
- [**5. Conclusion**](#5-conclusion)

<!-- /TOC -->

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

近两年，在computer vision领域，imagenet比赛前几名的，基本都是几十甚至上百层的网络，但NMT领域，成功的模型里，最深的也就6层。原因在于，**与卷积层相比，lstm里面有更多的非线性激活函数，而这些激活函数significantly decrease the magnititude（重要性）of the gradient in the deep topology, especially when the gradient progates in recurrent form.**

本文中使用了一种new type of **linear connections (fast forward connections)** for 多层的recurrent network。而且，我们还在**encoder中**使用了一个**interleaved bi-directional architecture to stack lstm layers**。这种拓扑可以在encoder-decoder网络中用，也可以在attention网络中使用。

# **2. NMT**

# **3. Deep Topology**

## **3.1 Network**

## **3.2 Train technique**

## **3.3 Generation**

# **4. Experiments**

## **4.1 Data sets**

## **4.2 Model settings**

## **4.3 Optimization**

## **4.4 Results**

### **4.4.1 Single models**

### **4.4.2 Post processing**

## **4.5 Analysis**

### **4.5.1 Length**

### **4.5.2 Unknown words**

### **4.5.3 Over-fitting**

# **5. Conclusion**
