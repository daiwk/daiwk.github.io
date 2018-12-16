---
layout: post
category: "knowledge"
title: "mmr"
tags: [mmr, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

参考

[https://github.com/jiangnanboy/NewsSummary](https://github.com/jiangnanboy/NewsSummary)

[https://blog.csdn.net/Silience_Probe/article/details/80700005](https://blog.csdn.net/Silience_Probe/article/details/80700005)

[https://blog.csdn.net/zjrn1027/article/details/81136761](https://blog.csdn.net/zjrn1027/article/details/81136761)

MMR是Maximal Marginal Relevance的缩写，中文为**最大边界相关算法**或**最大边缘相关算法**。

用来计算Query语句与被搜索文档之间的**相似度**，从而对文档进行rank排序的算法。

`\[
MMR(Q,C,R)= \underset{D_i \in R \setminus S}{argmax}[\lambda sim_1(Q,d_i)-(1-\lambda)\max _{D_j\in S}(sim_2(d_i, d_j))]
\]`

+ `\(D_i\)`：集合C中的一篇文档
+ Q: query
+ R: C中的相关文档集合
+ S: 当前结果集合
+ `\(\lambda\)`：可调超参，`\(\lambda\)`越大，准确率越高；`\(\lambda\)`越小，准确率越低

公式分为两部分，

+ 左边是候选句子和query的相似程度，
+ 右边部分是候选句子与所有已选择句子集合相似度最大值，取了负号，意味着**最终候选的句子间相似度越低越好**。



