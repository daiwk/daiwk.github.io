---
layout: post
category: "dl"
title: "nested lstm"
tags: [nested lstm,  ]
---

目录

<!-- TOC -->


<!-- /TOC -->

参考
[学界 \| Nested LSTM：一种能处理更长期信息的新型LSTM扩展](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650737297&idx=4&sn=075ed17c1fa9ec09309c1bea0f72785e&chksm=871aceefb06d47f9afd7fa28706f660c2dcaec3193575b7a637701ceb5620dcf7173975dcec9&mpshare=1&scene=1&srcid=0204MoahEmvMJaWVufDeThJS&pass_ticket=DtS40xhA8b%2FQB76bC%2FH86g91SmSrUAyY6MHLOfLSEdM7VjdptiHtx9tHknQ1s3BI#rd)

论文地址：

[Nested LSTMs](https://arxiv.org/pdf/1801.10308.pdf)

虽然在层级记忆上已有一些研究，LSTM 及其变体仍旧是处理时序任务最流行的深度学习模型，例如字符级的语言建模。特别是默认的堆栈 LSTM 架构使用一系列 LSTM 一层层地堆叠在一起来处理数据，一层的输出成为下一层的输入。在此论文中，研究者们提出并探索了一种全新的**嵌套 LSTM 架构（Nested LSTM，NLSTM)**，并认为其有潜力**直接取代堆栈 LSTM**。

在 NLSTM 中，LSTM 的**记忆单元**可以**访问内部记忆**，使用标准的 LSTM 门选择性地进行读取、编写。相比于传统的堆栈 LSTM，这一关键特征使得模型能**实现更有效的时间层级**。在 NLSTM 中，**(外部）记忆单元**可自由选择**读取、编写**的相关长期信息**到内部单元**。相比之下，在堆栈 LSTM 中，高层级的激活（类似内部记忆）直接生成输出，因此必须包含所有的与当前预测相关的短期信息。换言之，堆栈 LSTM 与嵌套 LSTM 之间的主要不同是，NLSTM 可以**选择性**地**访问内部记忆**。这使得**内部记忆**可以**免于记住、处理更长时间规模上的事件**，即使这些事件与当前事件不相关。

在此论文中，作者们的可视化图证明了，相比于堆栈 LSTM 中的高层级记忆，NLSTM 的**内部记忆**确实能**在更长的时间规模上操作**。实验也表明，NLSTM 在多种任务上都超越了堆栈 LSTM。

直观上，LSTM 中的**输出门**会编码**仍旧值得记忆的信息**，这些记忆可能与**当前的时间步**并**不相关**。嵌套 LSTM 根据这一直观理解来创造一种记忆的时间层级。访问**内部记忆**以同样的方式**被门控**，以便于长期信息只有在情景相关的条件下才能选择性地访问。

<html>
<br/>
<img src='../assets/nested lstm.webp' style='max-height: 300px'/>
<br/>
</html>

计算(outer)memory cell不是`\(c_t^{outer}=f_t\odot c_{t-1}+i_t\odot g_t\)`，而是将concatenation `\((f_t\odot c_{t-1}, i_t\odot g_t)\)`作为inner lstm(NLSTM)的memory cell的输入，并指定`\(c^{outer}_t=h^{inner}_t\)`。

