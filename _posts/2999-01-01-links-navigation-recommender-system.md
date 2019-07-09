---
layout: post
category: "links"
title: "【置顶】导引——推荐系统论文集合"
tags: [推荐系统, ]

---

目录

<!-- TOC -->

- [各种综述](#%E5%90%84%E7%A7%8D%E7%BB%BC%E8%BF%B0)
- [一些经典论文](#%E4%B8%80%E4%BA%9B%E7%BB%8F%E5%85%B8%E8%AE%BA%E6%96%87)
  - [dssm](#dssm)
  - [youtube](#youtube)
  - [序列建模](#%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1)
  - [标签体系](#%E6%A0%87%E7%AD%BE%E4%BD%93%E7%B3%BB)
- [最新paper1](#%E6%9C%80%E6%96%B0paper1)
  - [A review on deep learning for recommender systems: challenges and remedies](#A-review-on-deep-learning-for-recommender-systems-challenges-and-remedies)
  - [Next Item Recommendation with Self-Attention](#Next-Item-Recommendation-with-Self-Attention)
  - [Metric Factorization: Recommendation beyond Matrix Factorization](#Metric-Factorization-Recommendation-beyond-Matrix-Factorization)
  - [Collaborative Memory Network for Recommendation Systems](#Collaborative-Memory-Network-for-Recommendation-Systems)
  - [Evaluation of Session-based Recommendation Algorithms](#Evaluation-of-Session-based-Recommendation-Algorithms)
  - [RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems](#RippleNet-Propagating-User-Preferences-on-the-Knowledge-Graph-for-Recommender-Systems)
  - [Content-Based Citation Recommendation](#Content-Based-Citation-Recommendation)
  - [Explainable Recommendation: A Survey and New Perspectives](#Explainable-Recommendation-A-Survey-and-New-Perspectives)
  - [STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation](#STAMP-Short-Term-AttentionMemory-Priority-Model-for-Session-based-Recommendation)
  - [Real-time Personalization using Embeddings for Search Ranking at Airbnb](#Real-time-Personalization-using-Embeddings-for-Search-Ranking-at-Airbnb)
  - [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](#Billion-scale-Commodity-Embedding-for-E-commerce-Recommendation-in-Alibaba)
  - [Sequential Recommendation with User Memory Networks](#Sequential-Recommendation-with-User-Memory-Networks)
  - [Aesthetic-based Clothing Recommendation](#Aesthetic-based-Clothing-Recommendation)
  - [Multi-Pointer Co-Attention Networks for Recommendation](#Multi-Pointer-Co-Attention-Networks-for-Recommendation)
  - [ATRank: An Attention-Based User Behavior Modeling Framework for Recommendation](#ATRank-An-Attention-Based-User-Behavior-Modeling-Framework-for-Recommendation)
  - [Deep Matrix Factorization Models for Recommender Systems](#Deep-Matrix-Factorization-Models-for-Recommender-Systems)
- [最新paper2](#%E6%9C%80%E6%96%B0paper2)
  - [DeepRec: An Open-source Toolkit for Deep Learning based Recommendation](#DeepRec-An-Open-source-Toolkit-for-Deep-Learning-based-Recommendation)
  - [A Survey on Session-based Recommender Systems](#A-Survey-on-Session-based-Recommender-Systems)
  - [Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems](#Reinforcement-Learning-to-Optimize-Long-term-User-Engagement-in-Recommender-Systems)
  - [Representing and Recommending Shopping Baskets with Complementarity, Compatibility, and Loyalty](#Representing-and-Recommending-Shopping-Baskets-with-Complementarity-Compatibility-and-Loyalty)
  - [Attentive Group Recommendation](#Attentive-Group-Recommendation)
  - [Variational Autoencoders for Collaborative Filtering](#Variational-Autoencoders-for-Collaborative-Filtering)
  - [Exploiting Emotion on Reviews for Recommender Systems](#Exploiting-Emotion-on-Reviews-for-Recommender-Systems)
  - [An Attentive Interaction Network for Context-aware Recommendations](#An-Attentive-Interaction-Network-for-Context-aware-Recommendations)
  - [Regularizing Matrix Factorization with User and Item Embeddings for Recommendation](#Regularizing-Matrix-Factorization-with-User-and-Item-Embeddings-for-Recommendation)
  - [Explainable Recommendation via Multi-Task Learning in Opinionated Text Data](#Explainable-Recommendation-via-Multi-Task-Learning-in-Opinionated-Text-Data)
  - [Top-K Off-Policy Correction for a REINFORCE Recommender System](#Top-K-Off-Policy-Correction-for-a-REINFORCE-Recommender-System)
- [其他](#%E5%85%B6%E4%BB%96)
  - [Fast Matrix Factorization for Online Recommendation with Implicit Feedback](#Fast-Matrix-Factorization-for-Online-Recommendation-with-Implicit-Feedback)
- [2019顶会相关文章](#2019%E9%A1%B6%E4%BC%9A%E7%9B%B8%E5%85%B3%E6%96%87%E7%AB%A0)

<!-- /TOC -->

## 各种综述

传统mf算法：

[https://daiwk.github.io/posts/ml-recommender-systems.html](https://daiwk.github.io/posts/ml-recommender-systems.html)

其他几个综述：

[Batmaz2018\_Article\_AReviewOnDeepLearningForRecomm](https://daiwk.github.io/assets/Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf)

[Deep Learning based Recommender System: A Survey and New Perspectives](https://daiwk.github.io/assets/DL_on_RS.pdf)

[Deep Learning for Matching in Search and Recommendation](https://www.comp.nus.edu.sg/~xiangnan/sigir18-deep.pdf)

好像上面的链接403了，这个应该Ok：[http://www.hangli-hl.com/uploads/3/4/4/6/34465961/wsdm_2019_tutorial.pdf](http://www.hangli-hl.com/uploads/3/4/4/6/34465961/wsdm_2019_tutorial.pdf)

个人小结(更新中)：

传统部分：[https://daiwk.github.io/posts/dl-match-for-search-recommendation-traditional.html](https://daiwk.github.io/posts/dl-match-for-search-recommendation-traditional.html)

深度学习部分：[https://daiwk.github.io/posts/dl-match-for-search-recommendation.html](https://daiwk.github.io/posts/dl-match-for-search-recommendation.html)

## 一些经典论文

### dssm

[https://daiwk.github.io/posts/nlp-dssm.html](https://daiwk.github.io/posts/nlp-dssm.html)

### youtube

[https://daiwk.github.io/posts/dl-youtube-video-recommendation.html](https://daiwk.github.io/posts/dl-youtube-video-recommendation.html)

### 序列建模

[Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)

paddle实现：[https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/gru4rec](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/gru4rec)

### 标签体系

[https://daiwk.github.io/posts/nlp-tagspaces.html](https://daiwk.github.io/posts/nlp-tagspaces.html)

## 最新paper1

参考[想了解推荐系统最新研究进展？请收好这16篇论文](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247491818&idx=1&sn=311962e2e41119a565c252a19037dd76&chksm=96ea3f6aa19db67c3fbfa77fbec65797d0ccc8f2930290d57c2016a3e55a8bb18b77fd10180b&mpshare=1&scene=1&srcid=0928qyE33GaHUFg2ddzaDfmw&pass_ticket=QWrkCW0n7ulxJGBU0GG41I42RJFST5los9jWx8%2B%2BvEjJLwDxvQFM9Vs9mfvzCdFq#rd)


### A review on deep learning for recommender systems: challenges and remedies

本文是最新发表的一篇利用深度学习做推荐系统的综述，不仅从深度学习模型方面对文献进行了分类，而且从推荐系统研究的问题方面对文献做了分类。

### Next Item Recommendation with Self-Attention

本文提出了一种基于 self-attention 的基于序列的推荐算法，该算法是用 self-attention 从用户的交互记录中自己的去学习用的近期的兴趣，同时该模型也保留了用户的长久的兴趣。整个网络是在 metric learning 的框架下，是第一次将 self-attention 和 metric learning的结合的尝试。 

实验结果表明，通过 self-attention，模型可以很好的学习用户的短期兴趣爱好， 并且能有效的提升模型效果。通过和近期的文章得对比发现，该方法可以在很大程度上改善序列化推荐的效果。


### Metric Factorization: Recommendation beyond Matrix Factorization

本文提出了一种新型的推荐系统算法——Metric Factorization（距离分解）， 该方法旨在改进传统的基于矩阵分解的推荐系统算法。矩阵分解一个很大的问题就是不符合 inequality property， 这很大程度上阻碍了其表现。 

本文提出新型的解决方案，通过把用户和商品看作是一个低纬空间里面的点，然后用他们之间的距离来表示他们的距离。通过类似于矩阵分解的 squared loss 就能很好的从已有的历史数据中学出用户和商品在这个低维空间的位置。 

Metric Factorization 可以用在评分预测和排序两个经典的推荐场景，并且都取得了 state-of-the-art 的结果，超过基于深度学习以及已有的 Metric learning 的推荐算法。

[https://github.com/cheungdaven/metricfactorization](https://github.com/cheungdaven/metricfactorization)

### Collaborative Memory Network for Recommendation Systems

本文是圣塔克拉拉大学和 Google 联合发表于 SIGIR 2018 的工作。在传统的协同过滤模型中，隐藏因子模型能够捕捉交互的全局特征，基于近邻的相似度模型能够捕捉交互的局部特征。本文将两类协同过滤模型进行统一，根据注意力机制和记忆模块刻画复杂的用户-物品交互关系。

[https://github.com/tebesu/CollaborativeMemoryNetwork](https://github.com/tebesu/CollaborativeMemoryNetwork)

### Evaluation of Session-based Recommendation Algorithms

本文系统地介绍了 Session-based Recommendation，主要针对 baseline methods, nearest-neighbor techniques, recurrent neural networks, 和 (hybrid) factorization-based methods 等 4 大类算法进行介绍。

此外，论文使用 RSC15、TMALL、ZALANDO、RETAILROCKET、8TRACKS 、AOTM、30MUSIC、NOWPLAYING、CLEF 等 7 个数据集进行分析，在 Mean Reciprocal Rank (MRR)、Coverage、Popularity bias、Cold start、Scalability、Precision、Recall 等指标上进行比较。

[https://www.dropbox.com/sh/7qdquluflk032ot/AACoz2Go49q1mTpXYGe0gaANa?dl=0](https://www.dropbox.com/sh/7qdquluflk032ot/AACoz2Go49q1mTpXYGe0gaANa?dl=0)

### RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems

本文是上海交大、微软亚洲研究院和香港理工大学联合发表于 CIKM 2018 的工作。为了解决协同过滤的稀疏性和冷启动问题，研究人员通常利用社交网络或项目属性等辅助信息来提高推荐效果。本文将知识图谱应用到推荐系统中，是一个很新颖的方法，给推荐系统提供了一个全新的思路。

[https://github.com/hwwang55/RippleNet](https://github.com/hwwang55/RippleNet)

### Content-Based Citation Recommendation

本文提出了一种基于文章表示学习的方法，在为学术论文进行引文推荐任务上取得了较大成效。将给定的查询文档嵌入到向量空间中，然后将其邻近选项作为候选，并使用经过训练的判别模型重新排列候选项，以区分观察到的和未观察到的引用。此外，本文还发布了一个包含 700 万篇研究文章的公开数据集。

[https://github.com/allenai/citeomatic](https://github.com/allenai/citeomatic)

### Explainable Recommendation: A Survey and New Perspectives

本文是对“可解释性推荐系统”相关以及最新研究的调研总结，内容包括问题定义、问题历史、解决方案、相关应用和未来方向。论文内容较为全面，对于刚接触这一方向或者已经从事搭配领域的业者学者有很好的借鉴意义，文章最后对于一些可以发展的方向的论述也很有启发意义。

### STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation

本文是电子科大发表于 KDD 2018 的工作。论文提出了一种短期注意力/记忆优先的网络模型，在建模长时间序列的用户点击行为时，着重加强用户近期行为的影响。该方法既考虑了从长期历史行为挖掘用户的一般兴趣，又考虑了用户上一次点击挖掘用户的即时兴趣。实验表明，本文工作在 CIKM16 和 RSC15 两个经典数据集上均达到了最优结果。

### Real-time Personalization using Embeddings for Search Ranking at Airbnb

本文是 Airbnb 团队发表于 KDD 18 的工作，摘得 Applied Data Science Track Best Paper 奖项。论文介绍了 Airbnb 利用 word embedding 的思路训练 Listing（也就是待选择的民宿房间）和用户的 embedding 向量，并在此基础上实现相似房源推荐和实时个性化搜索。

### Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba

本文是阿里巴巴和香港科技大学发表于 SIGKDD 2018 的工作，论文结合节点 side information，解决了图表示学习中稀疏性和冷启动问题，在电商 i2i 推荐上取得很好的效果。

### Sequential Recommendation with User Memory Networks

本文是清华大学发表于 WSDM 2018 的工作。现有的深度学习推荐模型通常把用户的历史记录编码成一个 latent vector，但是可能会丢失 per-item 的共现信息。本文提出一种记忆增强的神经网络，把用户历史存到 memory 里，并设计了 read/write 机制更新 memory 内容，使序列网络能更动态地记录用户历史信息。

### Aesthetic-based Clothing Recommendation

本文是清华大学发表于 WWW 18 的工作，论文利用图片增强效果，传统的方法只考虑 CNN 抽取的图像特征；而本文考虑了图片中的美学特征对于推荐的影响；作者利用 BDN 从图片中学习美学特征，然后将其融合到 DCF 中，增强用户-产品，产品-时间矩阵，从而提高了推荐效果；在亚马逊和 AVA 数据集上都取得了良好的效果。

### Multi-Pointer Co-Attention Networks for Recommendation

[https://github.com/vanzytay/KDD2018_MPCN](https://github.com/vanzytay/KDD2018_MPCN)

本文是南洋理工大学发表于 KDD 2018 的工作。在预测用户对商品的评分时，如何学习用户和商品的表示至关重要。本文基于协同注意力机制，在 review-level 和 word-level 对用户评论和与商品相关的评论进行选择，选择最重要的一条或若干条评论来对用户和商品进行表示。

### ATRank: An Attention-Based User Behavior Modeling Framework for Recommendation

[https://github.com/jinze1994/ATRank](https://github.com/jinze1994/ATRank)

本文来自阿里巴巴，论文尝试设计和实现了一种能够融合用户多种时序行为数据的方法，较为创新的想法在于提出了一种同时考虑异构行为和时序的解决方案，并给出较为简洁的实现方式。使用类似 Google 的 self-attention 机制去除 CNN、LSTM 的限制，让网络训练和预测速度变快的同时，效果还可以略有提升。 此框架便于扩展。可以允许更多不同类型的行为数据接入，同时提供多任务学习的机会，来弥补行为稀疏性。

### Deep Matrix Factorization Models for Recommender Systems

[https://github.com/RuidongZ/Deep_Matrix_Factorization_Models](https://github.com/RuidongZ/Deep_Matrix_Factorization_Models)

本文在利用深度学习做推荐时，考虑了推荐的显式反馈和隐式反馈，将其融合构建成一个矩阵，从而将用户和产品的不同向量输入到两个并行的深层网络中去。最后，设计了一种新型的损失函数以同时考虑评分和交互两种不同类型的反馈数据。

## 最新paper2

参考[推荐系统阅读清单：最近我们在读哪些论文？](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247497070&idx=1&sn=f5d8489a9ae0edaf64281bb4e9e0f911&chksm=96ea2aeea19da3f81b5be9933497bdd921abd4dec0457fc49d0055a66d6d8f21c4e16ec1db0b&mpshare=1&scene=1&srcid=&pass_ticket=gsCKgAT5ncC4Oyqcqu32vlRROsqT7GGmlwcs6Mru0krjDac9tUZg2yxUB7dE5%2BpP#rd)

### DeepRec: An Open-source Toolkit for Deep Learning based Recommendation

[DeepRec: An Open-source Toolkit for Deep Learning based Recommendation](https://www.researchgate.net/publication/332555888_DeepRec_An_Open-source_Toolkit_for_Deep_Learning_based_Recommendation)

本文提出了一个全新开源工具库，包含诸多基于深度学习的推荐算法，代码可以直接运行，可以用来做为 baselines 和开发自己的算法，是一个非常不错的工具。

[https://github.com/cheungdaven/DeepRec](https://github.com/cheungdaven/DeepRec)

### A Survey on Session-based Recommender Systems

[A Survey on Session-based Recommender Systems](https://arxiv.org/abs/1902.04864)

本文是第一篇全面深入总结 session-based recommendations 的综述文章，值得推荐。文章系统总结了目前一种新型推荐范式：session-based recommendations 的特点、挑战和目前取得的进展，对整个推荐系统研究领域和相关的工业界人员提供了一个全面了解推荐系统领域最新研究进展的机会。

该文从问题本质和相关的数据特征入手，为 session-based recommendations 建立了一个层次化模型来深入理解里面存在的各种数据复杂性和潜在挑战，然后采用了两个不同维度对现有研究成果进行了系统分类和总结，最后提出了展望。

### Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems

[Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems](https://arxiv.org/abs/1902.05570)

论文针对利用强化学习解决推荐系统时存在用户行为难以建模的问题，提出了一种新的强化学习框架 FeedRec，包括两个网络：Q 网络利用层次化 LSTM 对复杂用户行为建模，S 网络用来模拟环境，辅助和稳定 Q 网络的训练。方法在合成数据和真实数据上进行了验证，取得了 SOTA 的结果。

### Representing and Recommending Shopping Baskets with Complementarity, Compatibility, and Loyalty

[Representing and Recommending Shopping Baskets with Complementarity, Compatibility, and Loyalty](https://www.microsoft.com/en-us/research/uploads/prod/2019/01/cikm18_mwan.pdf)

与在线购物不同的是，在超市购物场景下，商品间的互补性和用户对商品的忠诚度起着决定性作用。本文基于这两个维度提出了一种新的表示学习方法——triple2vec。此外，作者在上述方法得到的表示基础上，提出了一种考虑忠诚度的推荐算法，用忠诚系数来权衡表示模型和统计模型计算出的购买偏好。

源码：[https://github.com/MengtingWan/grocery](https://github.com/MengtingWan/grocery)

### Attentive Group Recommendation

[Attentive Group Recommendation](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir18-groupRS.pdf)

论文应用神经协同网络和注意力机制为群组用户进行 Top-N 商品推荐，主要解决了群组用户兴趣的动态组合、群组与个人用户的协同商品推荐，以及新用户的冷启动问题。

源码：[https://github.com/LianHaiMiao/Attentive-Group-Recommendation](https://github.com/LianHaiMiao/Attentive-Group-Recommendation)

### Variational Autoencoders for Collaborative Filtering

[Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814)

论文将变分自编码器（VAE）扩展到协同过滤以进行隐式反馈，通过非线性概率模型克服线性因子模型的局限。其次，作者引入了具有多项式似然（multinomial likelihood）的生成模型，并使用贝叶斯推断进行参数估计。作者基于 VAE 提出了一个生成模型 VAE_CF，并针对 VAE 的正则参数和概率模型选取做了适当调整，使其在当前推荐任务中取得最佳结果。

源码：[https://github.com/dawenl/vae_cf](https://github.com/dawenl/vae_cf)

### Exploiting Emotion on Reviews for Recommender Systems

[Exploiting Emotion on Reviews for Recommender Systems](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16608/16603)

现有的推荐系统大多依靠用户的历史评分或者评论文本进行推荐，往往由于数据资源不足而面临数据稀疏和难以进行冷启动的问题。本文基于原则性和数学的方式，对用户评论中的积极/消极情感加以充分利用，提出了一个全新推荐框架 MIRROR，并且在 Ciao 和 Epinions 这两个真实数据集上证明了该框架的有效性。

### An Attentive Interaction Network for Context-aware Recommendations

[An Attentive Interaction Network for Context-aware Recommendations](https://dl.acm.org/citation.cfm?id=3271813)

论文关注的问题是基于上下文感知的推荐系统。作者提出了一种新型注意力交互网络，用来捕捉内容、user和item之间的交互影响。此外，作者还提出了一种效应级注意力机制来聚合多种交互影响。通过在三个公开数据集 Food、Yelp 和 Frappe 上的大量实验表明，本文模型效果优于当前最先进的上下文感知推荐算法。

### Regularizing Matrix Factorization with User and Item Embeddings for Recommendation

[Regularizing Matrix Factorization with User and Item Embeddings for Recommendation](https://arxiv.org/pdf/1809.00979)

论文提出了一个基于 RME（Regularized Multi-Embedding）的推荐模型，不再基于用户的共同喜好进行物品推荐，而是创新地提出刻画物品对共同被讨厌的特征，进而避免向用户推荐其讨厌的物品。

源码：[https://github.com/thanhdtran/RME](https://github.com/thanhdtran/RME)

### Explainable Recommendation via Multi-Task Learning in Opinionated Text Data

[Explainable Recommendation via Multi-Task Learning in Opinionated Text Data](https://arxiv.org/abs/1806.03568)

论文提出了一个用于可解释推荐任务的多任务学习方法，通过联合张量分解将用户、产品、特征和观点短语映射到同一表示空间。

源码：[https://github.com/MyTHWN/MTER](https://github.com/MyTHWN/MTER)


### Top-K Off-Policy Correction for a REINFORCE Recommender System

[Top-K Off-Policy Correction for a REINFORCE Recommender System](https://arxiv.org/pdf/1812.02353.pdf)

参考[https://daiwk.github.io/posts/dl-topk-off-policy-correction.html](https://daiwk.github.io/posts/dl-topk-off-policy-correction.html)

## 其他

### Fast Matrix Factorization for Online Recommendation with Implicit Feedback

参考[https://daiwk.github.io/posts/ml-recommender-systems.html#eals](https://daiwk.github.io/posts/ml-recommender-systems.html#eals)

## 2019顶会相关文章

[最新！五大顶会2019必读的深度推荐系统与CTR预估相关的论文](https://zhuanlan.zhihu.com/p/69559974)

偷个图：

<html>
<br/>
<img src='../assets/recommender_system_dl.jpg' style='max-width: 300px'/>
<br/>
</html>
