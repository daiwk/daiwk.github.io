---
layout: post
category: "dl"
title: "federated learning/联邦学习"
tags: [federated learning, 联邦学习, ]
---

<!-- TOC -->

- [google2017年的blog](#google2017年的blog)
    - [paper1: Communication-Efficient Learning of Deep Networks from Decentralized Data](#paper1-communication-efficient-learning-of-deep-networks-from-decentralized-data)
    - [paper2: Federated Learning: Strategies for Improving Communication Efficiency](#paper2-federated-learning-strategies-for-improving-communication-efficiency)
    - [paper3: Federated Optimization: Distributed Machine Learning for On-Device Intelligence](#paper3-federated-optimization-distributed-machine-learning-for-on-device-intelligence)
    - [paper4: Practical Secure Aggregation for Privacy Preserving Machine Learning](#paper4-practical-secure-aggregation-for-privacy-preserving-machine-learning)
- [yangqiang的paper](#yangqiang的paper)

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

