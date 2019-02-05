---
layout: post
category: "dl"
title: "graph represention"
tags: [graph representation, ]
---

目录

<!-- TOC -->

- [Introduction](#introduction)
- [Node Representation Learning](#node-representation-learning)
    - [Node Representation Methods](#node-representation-methods)
        - [LINE](#line)
            - [一阶相似度](#一阶相似度)
            - [二阶相似度](#二阶相似度)
            - [优化trick](#优化trick)
            - [讨论](#讨论)
            - [实验](#实验)
        - [DeepWalk](#deepwalk)
        - [Node2vec](#node2vec)
    - [Graph and High-dimensional Data Visualization](#graph-and-high-dimensional-data-visualization)
    - [Knowledge Graph Embedding](#knowledge-graph-embedding)
    - [A High-performance Node Representation System](#a-high-performance-node-representation-system)
- [Graph Neural Networks](#graph-neural-networks)
    - [Graph Convolutional Networks](#graph-convolutional-networks)
    - [Graph Convolutional Networks](#graph-convolutional-networks-1)
    - [GraphSAGE](#graphsage)
    - [Gated Graph Neural Networks](#gated-graph-neural-networks)
    - [Graph Attention Networks](#graph-attention-networks)
    - [Subgraph Embeddings](#subgraph-embeddings)
- [Deep Generative Models for Graph Generation](#deep-generative-models-for-graph-generation)
    - [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)
    - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
    - [Deep Auto-regressive Models](#deep-auto-regressive-models)

<!-- /TOC -->


参考AAAI2019的tutorial：[AAAI2019《图表示学习》Tutorial, 180 页 PPT 带你从入门到精通（下载）](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652037887&idx=4&sn=730e6bd0cce6c3e40f4aec2de190f99d&chksm=f121960ec6561f1842640aff963fbe609e52fdd695386a442a9726016551c0a3697a196fd5a1&mpshare=1&scene=1&srcid=0202IYksgop7SmowGXrFkpMV&pass_ticket=IHFhi51iDztgrpluZGaofR7zoGSfaB%2F4Y6iACEc6lvxZ3KgGvbkA%2Fhp2MAVH09RS#rd)

ppt下载：[https://pan.baidu.com/s/1hRjm1nbMcj4_ynZ0niE2JA](https://pan.baidu.com/s/1hRjm1nbMcj4_ynZ0niE2JA)

传统的机器学习方法依赖于用户定义的启发式模型来提取关于图的结构信息的特征编码 (例如，degree statistics或核函数)。然而，近年来，使用基于深度学习和非线性降维的技术，自动学习将**图结构**编码为**低维embedding**的方法激增。

## Introduction

graph的几大传统ml任务：

+ **Node classification**：预测给定的结点的type
+ **Link prediction**：预测两个结点是否有边相连
+ **Community detection**：发现联系紧密的nodes的clusters
+ **Network similarity**：两个（子）网是否相似

目前的深度学习：

+ cnn：固定大小的图片/网格
+ rnn/w2v：文本/序列

图更加复杂：

+ 复杂的拓扑结构（例如，不像网格那样有spatial locality(空间局部性，在最近的将来将用到的信息很可能与现在正在使用的信息在**空间地址上是临近的**。)）
+ 没有固定的结点顺序或者参考点（reference point）(例如，isomorphism（同构）问题)
+ 经常是动态的并且有multimodal（多模态）的features

## Node Representation Learning

### Node Representation Methods

问题定义：给定`\(G=(V,E,W)\)`，其中，`\(V\)`是结点集合，`\(E\)`是边的集合，`\(W\)`是**边**的权重集合。所谓的node embedding就是对结点`\(i\)`学习一个向量`\(u_i\in R^d\)`。

相关工作：

+ 传统graph embedding算法：MDS, IsoMap, LLE, Laplacian Eigenmap, ...。缺点：hard to scale up
+ Graph factorization(Ahmed et al. 2013)：只适用于无向图，并非专门为网络表示而设计
+ Neural word embeddings(Bengio et al. 2003)：Neural language model；word2vec (skipgram), paragraph vectors, etc.

#### LINE

WWW2015上的[LINE: Large-scale Information Network Embedding](http://www.www2015.it/documents/proceedings/proceedings/p1067.pdf)

LINE代码（c++）：[https://github.com/tangjianpku/LINE](https://github.com/tangjianpku/LINE)

特点：

+ 任意类型的网络（有向图、无向图、有/无权重）
+ 明确的目标函数（一阶和二阶相似性（first/second proximity））
+ 可扩展性
    + 异步sgd
    + 百万级的结点和十亿级别的边：单机数小时

##### 一阶相似度

**First-order Proximity（一阶相似度）**：两个**顶点之间**的**自身**相似（不考虑其他顶点）。因为有些结点的link并没有被观测到，所以一阶相似度不足以保存网络结构。

分布：(定义在**无向**边`\(i-j\)`上)

一阶相似度的经验分布：

`\[
\hat{p_1}(v_i,v_j)=\frac{w_{ij}}{\sum_{(m,n)\in E}w_{mn}}
\]`

一阶相似度的模型分布：

`\[
p_1(v_i,v_j)=\frac{\exp(\vec{u_i}^T\vec{u_j})}{\sum_{(m,n)\in V\times V}\exp(\vec{u_m}^T\vec{u_n})}
\]`

其中，`\(\vec{u_i}\)`是节点`\(i\)`的embedding，其实就是sigmoid：

`\[
p_1(v_i,v_j)=\frac{1}{1+\exp(-\vec{u_i}^T\vec{u_j})}
\]`

目标函数是**KL散度**：

`\[
O_1=KL(\hat{p_1},p_1)
\]`

干掉常量`\(\sum_{(m,n)\in E}w_{mn}\)`，还有`\(\sum _{(i,j)\in E}w_{ij}\log w_{ij}\)`之后：

`\[
O_1=\sum _{(i,j)\in E}w_{ij}\log w_{ij}-\sum _{(i,j)\in E}w_{ij}\log p_1(v_i,v_j)\approx -\sum _{(i,j)\in E}w_{ij}\log p_1(v_i,v_j)
\]`

只考虑一阶相似度的情况下，改变同一条边的方向对于最终结果没有什么影响。因此一阶相似度只能用于**无向图**，不能用于有向图。

##### 二阶相似度

**Second-order Proximity（二阶相似度）**：网络中一对顶点`\((u,v)\)`之间的二阶相似度是它们**邻近网络结构**之间的相似性。

分布：(定义在**有向**边`\(i\rightarrow j\)`上)

邻近网络的经验分布：

`\[
\hat{p_2}(v_j|v_i)=\frac{w_{ij}}{\sum_{k\in V}w_{ik}}
\]`

邻近网络的模型分布，其中，`\(u_i\)`是`\(v_i\)`被视为顶点时的表示，`\(u'_i\)`是`\(v_i\)`被视为"context"时的表示：

`\[
p_2(v_j|v_i)=\frac{\exp(\vec{u'_j}^T\vec{u_i})}{\sum_{k\in V}\exp(\vec{u'_k}^T\vec{u_i})}
\]`

目标函数是**KL散度**：

`\[
O_2=\sum_i KL(\hat{p_2}(\cdot |v_i),p_2(\cdot|v_i))=-\sum _{(i,j)\in E}w_{ij}\log p_2(v_j|v_i)
\]`

##### 优化trick

+ sgd+negative sampling：随机sample一条边，以及多个negative的边

例如针对二阶的，对每条边`\((i,j)\)`来说，它的目标函数就是：

`\[
\log \sigma(\vec{u'_j}^T\vec{u'_i})+\sum ^K_{i=1}E_{v_n\sim P_n(v)}[\log \sigma (-\vec{u'_n}^T\vec{\u_i})]
\]`

其中`\(\sigma(x)=1/(1+\exp(-x))\)`，设置`\(P_n(v)\propto d_v^{3/4}\)`，其中`\(d_v\)`是节点的出度（即`\(d_i=\sum _{k\in N(i)}w_{ik}\)`，其中`\(N(i)\)`是`\(v_i\)`的为起点的邻居的集合）。

针对一阶的，把上面式子里的第一项里的`\(\vec{u'_j}^T\)`换成`\(\vec{u_j}\)`就行啦~

+ 边`\((i,j)\)`的embedding的梯度：

`\[
\frac{\partial O_2}{\partial \vec{u_i}}=w_{ij}\frac{\partial \log \hat{p_2}(v_j|v_i)}{\partial \vec{u_i}}
\]`

+ 当边的权重方差很大的时候，从上式可知，目标函数的梯度是`\(p_2\)`的梯度再乘以边权重，所以目标函数的梯度的方差也会很大，这样会有问题。
+ 解决方法：**edge sampling**：根据边的权重来采样边，然后把采样到的边当成binary的，也就是把每条边的权重看成一样的！(例如一个边的权重是`\(w\)`，那么拆成`\(w\)`条binary的边)
+ 复杂度：`\(O(d\times K \times |E|)\)`：`\(d\)`是embedding的维数，`\(K\)`是负样本的个数，`\(|E|\)`是边的总数

##### 讨论

+ 对只有少量邻居的节点（low degree vertices）进行embed：
    + 通过增加高阶邻居来扩展邻居
    + BFS(breadth-first search)，使用广度优先搜索策略扩展每个顶点的邻域，即递归地添加邻居的邻居
    + 在大部分场景下，只增加二阶邻居就足够了
+ 对新节点进行emb（如果新节点和已有节点有边相连，可以如下方式来搞；否则，future work...）:
    + 保持现有节点的embedding不变
    + 根据新节点的embedding求经验分布和模型分布，从而优化目标函数 w.r.t. 新node的embedding

所以，对于新节点`\(i\)`，直接最小化如下目标函数：

`\[
-\sum_{j\in N(i)}w_{ji}\log p_1(v_j,v_i)
\]`

或者

`\[
-\sum _{j\in N(i)}w_{ji}\log p_2(v_j|v_i)
\]`

##### 实验

LINE(1st)只适用于无向图，LINE(2nd)适用于各种图。

LINE (1st+2nd)：同时考虑一阶相似度和二阶相似度。将由LINE（1st）和LINE（2nd）学习得到的两个向量表示，**连接成一个更长的向量**。在连接之后，对维度**重新加权**以**平衡两个表示**。因为在无监督的任务中，设定权重很困难，所以**只应用于监督学习**的场景。

更适合的方法是共同训练一阶相似度和二阶相似度的目标函数，比较复杂，文章中没有实现。

#### DeepWalk

KDD14上的[DeepWalk: Online Learning of Social Representations](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)

使用学习word representation的方法来学习node representation（例如skip gram）

将网络上的随机游走视为句子。

分成两步：

+ 通过随机游走生成结点的context
+ 预测周围的节点:

`\[
p(v_j|v_i)=\frac{\exp(\vec{u'_i}^T \vec{u_j})}{\sum _{k\in V}\exp(\vec{u'_k}^T\vec{u_i})}
\]`

#### Node2vec

KDD16上的[node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf)

通过如下混合策略去寻找一个node的context：

+ Breadth-firstSampling(BFS): homophily（同质性）
+ Depth-firstSampling(DFS): structuralequivalence（结构等价）

使用带有参数`\(p\)`和`\(q\)`的**Biased Random Walk**来进行context的扩展：

+ `\(p\)`：控制在walk的过程中，revisit一个节点的概率
+ `\(q\)`：控制探索"outward"节点的概率
+ 在有标签的数据上，用cross validation来寻找最优的`\(p\)`和`\(q\)`

优化目标和LINE的**一阶相似度**类似

LINE、DeepWalk、Node2vec的对比：

html>
<center>
<table border="2" cellspacing="0" cellpadding="6" rules="all" frame="border">

<thead>
<tr>
<th scope="col" class="left">algorithm</th>
<th scope="col" class="left">neighbor expansion</th>
<th scope="col" class="left">proximity</th>
<th scope="col" class="left">optimization</th>
<th scope="col" class="left">validation data</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left">LINE</td>
<td class="left">BFS</td>
<td class="left">1st or 2nd</td>
<td class="left">negative sampling</td>
<td class="left">No</td>
</tr>
<tr>
<td class="left">DeepWalk</td>
<td class="left">Random</td>
<td class="left">2nd</td>
<td class="left">hierarchical softmax</td>
<td class="left">No</td>
</tr>
<tr>
<td class="left">Node2vec</td>
<td class="left">BFS+DFS</td>
<td class="left">1st</td>
<td class="left">negative sampling</td>
<td class="left">Yes</td>
</tr>

</tbody>
</table></center>
</html>


### Graph and High-dimensional Data Visualization

largevis代码（c++&python）：[https://github.com/lferry007/LargeVis](https://github.com/lferry007/LargeVis)

### Knowledge Graph Embedding

### A High-performance Node Representation System

RotatE代码（pytorch）:[https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding]

## Graph Neural Networks

### Graph Convolutional Networks

### Graph Convolutional Networks

### GraphSAGE

### Gated Graph Neural Networks

### Graph Attention Networks

### Subgraph Embeddings

## Deep Generative Models for Graph Generation

### Variational Autoencoders (VAEs)

### Generative Adversarial Networks (GANs)

### Deep Auto-regressive Models




