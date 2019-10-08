---
layout: post
category: "rl"
title: "rl for recommendation"
tags: [rl recommendation, 强化学习 推荐, drn, slateq, topk off policy, dear, ]
---

目录

<!-- TOC -->

- [drn](#drn)
- [top-k off-policy](#top-k-off-policy)
- [slateq](#slateq)
- [DEAR](#dear)

<!-- /TOC -->

## drn

[https://daiwk.github.io/posts/rl-drn.html](https://daiwk.github.io/posts/rl-drn.html)

## top-k off-policy

[https://daiwk.github.io/posts/dl-topk-off-policy-correction.html](https://daiwk.github.io/posts/dl-topk-off-policy-correction.html)

## slateq

[https://daiwk.github.io/posts/rl-slateq.html](https://daiwk.github.io/posts/rl-slateq.html)

## DEAR

[今日头条最新论文，首次改进DQN网络解决推荐中的在线广告投放问题](https://zhuanlan.zhihu.com/p/85417314)

[Deep Reinforcement Learning for Online Advertising in Recommender Systems](https://arxiv.org/abs/1909.03602)

在给定推荐列表前提下，本文提出了一种基于DQN的创新架构来同时解决三个任务：是否插入广告；如果插入，插入哪一条广告；以及插入广告在推荐列表的哪个位置。

DQN的两种经典结构：

+ 接受的输入是state，输出是所有可能action对应的Q-value；
+ 接受的输入是state以及某一个action，输出是对应的Q-value。

这两种经典架构的最主要的问题是只能将action定义为插入哪一条广告，或者插入广告在列表的哪个位置，无法同时解决上述提到的三个任务。

融合了上述提到了两种经典DQN结构的结合，输入层包含State以及Action（插入哪条广告），输出层则是广告插入推荐列表的L+1位置对应的Q-value（假设推荐列表长度为L，则可以插入广告的位置为L+1种可能）。与此同时，使用一个特殊插入位置0用来表示不进行广告插入，因此输出层的长度扩展成为L+2。

输出层Q函数被拆解成两部分：只由state决定的V函数；以及由state和action同时决定的A函数。其中，

+ state包含了使用GRU针对推荐列表和广告进行用户序列偏好建模的p；当前用户请求的上下文信息c；以及当前请求展示的推荐列表item的特征进行拼接转换形成的低维稠密向量rec；
+ action则包含两部分：一部分是候选插入广告ad的特征；另一部分则是广告插入的位置；其中这里的前半部分会被当做输入层。
+ reward函数。Reward函数也包含两部分：一部分是广告的的收入r^ad；另一部分则是用户是否继续往下刷的奖励。基于下图的reward函数，最优的Q函数策略便可以通过Bellman等式求得。


<html>
<br/>
<img src='../assets/dear-arch.png' style='max-height: 250px'/>
<br/>
</html>

基于用户交互历史的离线日志，采用 Off-policy的方式进行训练得到最优的投放策略。


针对每一次迭代训练：

+ （第6行）针对用户请求构建state；
+ （第7行）根据标准的off-policy执行action，也就是选取特定ad；
+ （第8行）根据设计好的reward函数，计算reward；
+ （第10行）将状态转移信息（s_t，a_t，r_t，s_t+1）存储到replay buffer；
+ （第11行）从replay buffer中取出mini-batch的状态转移信息，来训练得到最优的Q函数参数。

<html>
<br/>
<img src='../assets/off-policy-dear.png' style='max-height: 250px'/>
<br/>
</html>