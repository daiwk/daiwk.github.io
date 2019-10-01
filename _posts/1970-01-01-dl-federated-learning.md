---
layout: post
category: "dl"
title: "federated learning/联邦学习"
tags: [federated learning, 联邦学习, ]
---

<!-- TOC -->

- [google2017年的blog](#google2017%e5%b9%b4%e7%9a%84blog)
  - [paper1: Communication-Efficient Learning of Deep Networks from Decentralized Data](#paper1-communication-efficient-learning-of-deep-networks-from-decentralized-data)
  - [paper2: Federated Learning: Strategies for Improving Communication Efficiency](#paper2-federated-learning-strategies-for-improving-communication-efficiency)
  - [paper3: Federated Optimization: Distributed Machine Learning for On-Device Intelligence](#paper3-federated-optimization-distributed-machine-learning-for-on-device-intelligence)
  - [paper4: Practical Secure Aggregation for Privacy Preserving Machine Learning](#paper4-practical-secure-aggregation-for-privacy-preserving-machine-learning)
- [yangqiang的paper](#yangqiang%e7%9a%84paper)
- [FATE](#fate)
- [近期进展](#%e8%bf%91%e6%9c%9f%e8%bf%9b%e5%b1%95)

<!-- /TOC -->

## google2017年的blog

[https://ai.googleblog.com/2017/04/federated-learning-collaborative.html](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)

联邦学习能不用把数据从mobile传到cloud就可以训模型，也就是说mobile不仅可以做prediction，还可以搞训练。

大致流程如下，首先一个手机把当前模型下载下来，然后使用本机的数据进行更新，并把这些变化summarize成a small focused update。只有这个update会通过加密上传到cloud，然后这个用户的update会和其他用户的update一起进行平均，拿来更新shared model。所有的训练数据都存在设备上，不会把个人的update存到cloud去。

<html>
<br/>
<img src='../assets/FederatedLearning_FinalFiles_Flow Chart1.png' style='max-width: 300px'/>
<br/>
</html>

这种方式也有一个immediate benefit，那就是，在给shared model提供update之外，本地的improved model也可以立即在本地生效。

传统的机器学习系统，在cloud中通过partition分布在异构server上的大数据集上，使用sgd等优化算法。这种高度iterative的算法，要求与训练数据有低延迟、高吞吐的连接。而在federal learning的场景下，明显是更高的延迟、更低的吞吐，而且拿来训练的时候是intermittently（断断续续的）。

因此有了Federated Averaging algorithm，也就是[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)，能够使用比naively federated version的sgd少10-100倍的连接来训练深度网络。核心思想是使用手机上更强大的处理器，计算比普通的gradient steps更higher quality的update。这样，因为用了high quality的update，所以通过更少的迭代就可以得到一个好的模型了，所以需要的连接也更少了。

因为上传速度往往比下载速度慢得多(可以参考[https://www.speedtest.net/reports/united-states/](https://www.speedtest.net/reports/united-states/))，可以采用random rotation和quantization来得到压缩后的updates，能够降低100倍的上传耗时，参考[Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492)。

而对于高维稀疏的凸模型，例如ctr预估问题，可以参考[Federated Optimization: Distributed Machine Learning for On-Device Intelligence](https://arxiv.org/abs/1610.02527)。

部署需要一个复杂的技术栈，on device training需要一个miniature（小型的）版本的tf，调度需要仔细的设计，保证当设备处于idle、充电中、有wifi等场景下，保证对用户的体验没有影响。

通信与聚合要求是一种secure, efficient, scalable, and fault-tolerant的方式，所以只有包括了这些的infrastructure可以被考虑使用。

参考[https://eprint.iacr.org/2017/281](https://eprint.iacr.org/2017/281)，提出了[Practical Secure Aggregation for Privacy Preserving Machine Learning](https://eprint.iacr.org/2017/281.pdf)，也就是一种Secure Aggregation protocol，一个coordinating server只有当100s或者1000s的用户参与了，才会对average update进行解密，因此在average之前，没有哪个个人手机的update能被观察(inspected)到。所以coordinating server只需要average update，使得Secure Aggregation能被使用。不过这个协议还没有实现出来，有希望在near future得以部署。

目前的研究还只是皮毛，联邦学习解决不了不是所有机器学习问题，例如，[https://ai.googleblog.com/2016/08/improving-inception-and-image.html](https://ai.googleblog.com/2016/08/improving-inception-and-image.html)提到的在carefully标注的数据集上识别不同狗的种类的问题，或者是很多训练数据早已存储在cloud的问题(例如训练垃圾邮件识别问题)。所以，cloud-based的ml会持续发展，而federated learning也会持续寻求解决更多问题。

### paper1: Communication-Efficient Learning of Deep Networks from Decentralized Data

[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

[http://column.hongliangjie.com/%E8%AF%BB%E8%AE%BA%E6%96%87/2017/06/18/aistats2017-dist-sgd/](http://column.hongliangjie.com/%E8%AF%BB%E8%AE%BA%E6%96%87/2017/06/18/aistats2017-dist-sgd/)

FederatedAveraging算法总共有三个基本的参数，`\(C\)`（0到1）控制有多少比例的的客户端参与优化，`\(E\)`控制每一轮多少轮SGD需要在客户端运行，`\(B\)`是每一轮的Mini-Batch的数目大小。另外，假设总共有`\(K\)`个客户端。

+ 每一轮都**随机选择**出`\(\max (CK,1)\)`个的客户端
+ 对于**每个客户端**进行**Mini-Batch**的大小为`\(B\)`，**轮数**为`\(E\)`的SGD更新
+ 对于参数直接进行**加权平均**（这里的**权重**是每个**客户端的数据相对大小**）

文章对这里的最后一步进行了说明。之前有其他研究表明，如何直接对参数空间进行加权平均，特别是Non-Convex的问题，会得到任意坏的结果。这篇文章里，作者们对于这样的问题的处理是，让**每一轮**的**各个客户端**的**起始参数值相同**（也就是**前一轮的全局参数值**）。这一步使得算法效果大幅度提高。

### paper2: Federated Learning: Strategies for Improving Communication Efficiency

[Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492)

使用两种update来减少通信的耗时：

+ structured updates: 在一个使用**更少的变量**（例如low-rank或者随机mask）来参数化表示的**受限空间内**，**直接学习**一个**update**。
+ sketched updates: 学习**整个模型的update**，并在发送给server之前，使用**quantization, random rotations, and subsampling**来进行**压缩**。


### paper3: Federated Optimization: Distributed Machine Learning for On-Device Intelligence

[Federated Optimization: Distributed Machine Learning for On-Device Intelligence](https://arxiv.org/abs/1610.02527)

提到了SVRG(Stochastic Variance Reduced Gradient)、 DANE(Distributed Approximate Newton algorithm)，然后提出了Federated SVRG(FSVRG)。

### paper4: Practical Secure Aggregation for Privacy Preserving Machine Learning

[Practical Secure Aggregation for Privacy Preserving Machine Learning](https://eprint.iacr.org/2017/281.pdf)

## yangqiang的paper

[Federated Machine Learning: Concept and Applications](https://arxiv.org/pdf/1902.04885.pdf)

## FATE

[怎样扩充大数据？你需要了解的第一个联邦学习开源框架FATE](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650765998&idx=4&sn=a6fdc4c39e29e0260dc06779bceed6ad&chksm=871abed0b06d37c617a063ba4c98867658f981b4fae6d34e0f2785edebeae8a09b6a2cef22e7&mpshare=1&scene=1&srcid=&pass_ticket=zzUnWIgdqTLvX39vSLCKaOJN8KVDYuvxPgj7h5mQNNMiTnEMdrWSwBJSd3ch3aLL#rd)

github：[https://github.com/WeBankFinTech/FATE](https://github.com/WeBankFinTech/FATE)

## 近期进展

[打破数据孤岛：联邦学习近期重要研究进展](https://mp.weixin.qq.com/s/s4E9jM_HmOf9G4m0TIy61Q)

经典的联邦学习问题基于存储在数千万至数百万远程客户端设备上的数据学习全局模型。在训练过程中，客户端设备需要周期性地与中央服务器进行通信。目前，联邦学习面临的难点主要包括四个方面：

+ 高昂的通信代价。在联邦学习问题中，原始数据保存在远程客户端设备本地，必须与中央服务器不断交互才能完成全局模型的构建。通常整个联邦学习网络可能包含了大量的设备，网络通信速度可能比本地计算慢许多个数量级，这就造成高昂的通信代价成为了联邦学习的关键瓶颈。
+ 系统异质性。由于客户端设备硬件条件（CPU、内存）、网络连接（3G、4G、5G、WiFi）和电源（电池电量）的变化，联邦学习网络中每个设备的存储、计算和通信能力都有可能不同。网络和设备本身的限制可能导致某一时间仅有一部分设备处于活动状态。此外，设备还会出现没电、网络无法接入等突发状况，导致瞬时无法连通。这种异质性的系统架构影响了联邦学习整体策略的制定。
+ 统计异质性。设备通常以不同分布方式在网络上生成和收集数据，跨设备的数据数量、特征等可能有很大的变化，因此联邦学习网络中的数据为非独立同分布（Non-indepent and identically distributed, Non-IID）的。目前，主流机器学习算法主要是基于 IID 数据的假设前提推导建立的。因此，异质性的 Non-IID 数据特征给建模、分析和评估都带来了很大挑战。
+ 隐私问题。联邦学习共享客户端设备中的模型参数更新（例如梯度信息）而不是原始数据，因此在数据隐私保护方面优于其他的分布式学习方法。然而，在训练过程中传递模型的更新信息仍然存在向第三方或中央服务器暴露敏感信息的风险。隐私保护成为联邦学习需要重点考虑的问题。

4篇paper：

+ [Client selection for federated learning with heterogeneous resources in mobile edge](https://arxiv.org/abs/1804.08333), 提出了一个用于机器学习的移动边缘计算框架，它利用分布式客户端数据和计算资源来训练高性能机器学习模型，同时保留客户端隐私；
+ [Agnostic Federated Learning](https://arxiv.org/abs/1902.00146v1)，解决之前联邦学习机制中会对某些客户端任务发生倾斜的问题；
+ [Bayesian Nonparametric Federated Learning of Neural Networks](https://arxiv.org/abs/1905.12022v1). ICML 2019. 提出单样本/少样本探索式的学习方法来解决通信问题；
+ [Protection Against Reconstruction and Its Applications in Private Federated Learning](https://arxiv.org/pdf/1812.00984.pdf)，提出了一种差异性隐私保护方法。

