---
layout: post
category: "dl"
title: "parameter server详解"
tags: [parameter server, pserver ]
---

目录

<!-- TOC -->

- [1. 背景](#1-背景)
- [2. 发展历程](#2-发展历程)
- [3. 对比parameter server与通用分布式系统](#3-对比parameter-server与通用分布式系统)
- [4. parameter server的优势](#4-parameter-server的优势)
- [5. parameter server系统架构](#5-parameter-server系统架构)

<!-- /TOC -->

[Parameter Server 详解](http://blog.csdn.net/cyh_24/article/details/50545780)

参考论文：
[Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)

## 1. 背景

现实中，训练数据的数量可能达到1TB到1PB之间，而训练过程中的参数可能会达到`\(10^9\)`（十亿）到`\(10^12\)`（千亿）。而往往这些模型的参数需要被所有的worker节点频繁的访问，就有如下问题与挑战：

+ 需要大量的网络带宽支持
+ 很多机器学习算法都是连续型的，**只有上一次迭代完成（各个worker都完成）之后，才能进行下一次迭代，**这就导致了如果机器之间性能差距大（木桶理论），就会造成性能的极大损失；
+ 在分布式中，**容错能力**是非常重要的。很多情况下，算法都是部署到云环境中的（这种环境下，机器是不可靠的，并且job也是有可能被抢占的）

## 2. 发展历程

+ 第一代 parameter server：缺少灵活性和性能 —— 仅使**用memcached(key, value) 键值对存储作为同步机制。**YahooLDA 通过改进这个机制，增加了一个专门的服务器，提供用户能够自定义的更新操作(set, get, update)。 
+ 第二代 parameter server：用**bounded delay模型**来改进YahooLDA，但是却进一步限制了worker线程模型。 
+ 第三代 parameter server 能够解决这些局限性。

## 3. 对比parameter server与通用分布式系统

通用的分布式系统通常都是：每次迭代都**强制同步**，通常在几十个节点上，它们的性能可以表现的很好，但是在**大规模集群中，这样的每次迭代强制同步的机制会因为木桶效应变得很慢。**

**Mahout 基于 Hadoop，MLI 基于 Spark，**它们（Spark与MLI）采用的都是 **Iterative MapReduce 的架构**。它们能够保持迭代之间的状态，并且执行策略也更加优化了。但是，由于这两种方法都采用**同步迭代的通信方式**，使得它们很容易因为个别机器的低性能导致全局性能的降低。

为了解决这个问题，**Graphlab 采用图形抽象的方式进行异步调度通信。**但是它**缺少了以 MapReduce 为基础架构的弹性扩展性**，并且它使用**粗粒度的snapshots来进行恢复，这两点都会阻碍到可扩展性。**parameter server 正是吸取Graphlab异步机制的优势，并且解决了其在可扩展性方面的劣势。

## 4. parameter server的优势

+ Efficient communication
由于是**异步的通信**，因此，不需要停下来等一些机器执行完一个iteration（除非有必要），这大大减少了延时。为机器学习任务做了一些优化(后续会细讲)，能够大大减少网络流量和开销
+ Flexible consistency models
**宽松的一致性要求**进一步减少了同步的成本和延时。parameter server 允许算法设计者根据自身的情况来做算法收敛速度和系统性能之间的trade-off。
+ Elastic Scalability
使用了一个**分布式hash表**使得**新的server节点可以随时动态的插入到集合中**；因此，**新增一个节点不需要重新运行系统。**
+ Fault Tolerance and Durability
节点故障是不可避免的，特别是在大规模商用服务器集群中。**从非灾难性机器故障中恢复，只需要1秒，而且不需要中断计算**。**Vector clocks**保证了经历故障之后还是能运行良好
+ Ease of Use
**全局共享的参数**可以被表示成各种形式：vector，matrices 或者相应的sparse类型，这大大方便了机器学习算法的开发。并且提供的线性代数的数据类型都具有高性能的多线程库。

## 5. parameter server系统架构

在parameter server中，每个 server 实际上都只负责分到的**部分参数**（servers共同维持一个全局的共享参数），而每个 work 也只分到**部分数据**和处理任务。


