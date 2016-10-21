---
layout: post
category: "nlp"
title: "lstm crf"
tags: [nlp, natural language processing, lstm crf, lstm, crf]
---

# **1. CRF**

CRF(Conditional Random Field)条件随机场是用于标注和划分**序列结构数据**的**概率化**结构模型。

对于给定的输出标识序列Y和观测序列X，CRF通过定义**条件概率`\(P(Y|X)\)`**而非联合概率`\(P(X,Y)\)`来描述模型。也可以将CRF看作一个**无向图模型**或者，马尔可夫随机场（Markov Random Field）。

定义：设`\(G=(V,E)\)`为一个无向图，`\(V\)`为结点集合，`\(E\)`为无向边的集合。`\(Y=\{Y_v|v\in V\}\)`，即`\(V\)`中的 每一个结点对应于一个随机变量`\(Y_v\)`，其取值范围为可能的标记集合`\(\{y\}\)`。如果以观察序列`\(X\)`为条件，每一个随机变量`\(Y_v\)`都满足以下马尔可夫特性`(\eqref{Eq-1.1}\)`：

`\[
p(Y_v|X,Y_w,w\neq v)=P(Y_v|X,Y_w,w\sim v)
\tag{Eq-1.1}
\label{eq:sample}
\]`



# **2. 【2015】Bidirectional LSTM-CRF Models for Sequence Tagging**

# **3. 【2016】Conditional Random Fields as Recurrent Neural Networks**
 
