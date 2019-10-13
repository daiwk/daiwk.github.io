---
layout: post
category: "dl"
title: "图神经网络+推荐"
tags: [gnn, gcn, 推荐, graph, 图神经网络, 图网络, DiffNet, DGRec, IGPL, GraphRec, DANSER, NGCF]
---

目录

<!-- TOC -->

- [DiffNet](#diffnet)
- [DGRec](#dgrec)
- [IGPL](#igpl)
- [GraphRec](#graphrec)
- [DANSER](#danser)
- [NGCF](#ngcf)
- [MEIrec](#meirec)

<!-- /TOC -->

参考[近期必读的6篇【图神经网络的推荐（GNN+R）】相关论文和代码（WWW、SIGIR、WSDM）](https://mp.weixin.qq.com/s?__biz=MzU2OTA0NzE2NA==&mid=2247510850&idx=1&sn=222f99d740acd50bcbcf5a5745c5d938&chksm=fc864a51cbf1c34739a348177e109d4ff4323f313208c410a44ac6af4cc4e32ad66f6275b235&mpshare=1&scene=1&srcid=&pass_ticket=yi5ku1%2Fs0oomHKKecAzMcpWLxtfI6PDYKYJn%2BWzsyCs3SOPlL0ZxjXuFZUZ2FS5h#rd)

参考[CIKM 2019 EComm AI：用户行为预测 赛题解读与阿里GNN推荐结合实践分享](https://tianchi.aliyun.com/course/video?spm=5176.12586971.1001.62.55e2194dAHpQl4&liveId=41071)


## DiffNet

SIGIR ’19，

[A Neural Influence Diffusion Model for Social Recommendation](https://arxiv.org/pdf/1904.10322.pdf)

社交推荐系统利用每个用户的局部邻居偏好(local neighbors’ preferences)来缓解数据稀疏性，从而更好地进行用户emb建模。对于每一个社交平台的用户，其潜在的嵌入是受他信任的用户影响的，而这些他信任的用户也被他们自己的社交联系所影响。随着社交影响在社交网络中递归传播和扩散（diffuse），每个用户的兴趣在递归过程中发生变化。然而，目前的社交推荐模型只是利用每个用户的**局部邻居**来构建**静态模型**，没有模拟全局社交网络中的**递归扩散**，导致推荐性能不理想。

本文提出了一个deep influence propagation model。对于每个用户，扩散过程（diffusion）用融合了相关特征和一个caputure了latent behavior preference的free的用户隐向量。本文的key idea是，设计了一个layer-wise的influence propagation结构，可以随着social diffusion process的进行，对用户emb进行演化。

<html>
<br/>
<img src='../assets/diffnet.png' style='max-height: 300px'/>
<br/>
</html>

## DGRec

WSDM ’19，

[Session-based Social Recommendation via Dynamic Graph Attention Networks](https://arxiv.org/abs/1902.09362v2)

代码：[https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec)


## IGPL

[Inductive Graph Pattern Learning for Recommender Systems Based on a Graph Neural Network](https://arxiv.org/abs/1904.12058v1)


## GraphRec

WWW'19，

[Graph Neural Networks for Social Recommendation](https://arxiv.org/pdf/1902.07243.pdf)


## DANSER

WWW'19 Oral

[Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender Systems](https://arxiv.org/abs/1903.10433)

## NGCF

SIGIR'19

[Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108)

参考[https://www.jianshu.com/p/16c8973ef8ff](https://www.jianshu.com/p/16c8973ef8ff)

## MEIrec

[节后收心困难？这15篇论文，让你迅速找回学习状态](https://mp.weixin.qq.com/s/aaz-s87vorroyepNCd9-AA)

[Metapath-guided Heterogeneous Graph Neural Network for Intent Recommendation](http://shichuan.org/doc/67.pdf)

本文是北京邮电大学和阿里巴巴发表于 KDD 2019 的工作。针对手机淘宝的用户意图推荐，本文设计了基于异质图神经网络的意图推荐模型 MEIRec。

传统商品推荐为用户推荐商品，而意图推荐则关注于预测用户的意图。本文将意图推荐的业务场景建模为异质图（包含多种类型节点和关系的图），然后设计了 metapath-guided heterogeneous Graph Neural Network 来学习该业务场景下多种不同目标的表示。同时，本文也提出一种 term embedding mechanism 来降低大规模异质图场景下的参数量。最后，在淘宝真实场景下的 AB test 证明了 MEIRec 算法的优越性。

[https://github.com/googlebaba/KDD2019-MEIRec](https://github.com/googlebaba/KDD2019-MEIRec)
