---
layout: post
category: "dl"
title: "meta learning& auto-ml"
tags: [meta learning, learning to learn, automl, auto-ml, auto-gan, autogan, fairnas, efficientnet, adanet, mdenas, proxylessnas, haq, once for all, ]
urlcolor: blue
---


<!-- TOC -->

- [meta-learning](#meta-learning)
- [auto-ml](#auto-ml)
  - [fairnas](#fairnas)
  - [efficientnet](#efficientnet)
  - [adanet](#adanet)
  - [mdenas](#mdenas)
  - [ProxylessNAS & HAQ](#proxylessnas--haq)
  - [进化论方法](#%e8%bf%9b%e5%8c%96%e8%ae%ba%e6%96%b9%e6%b3%95)
  - [autogan](#autogan)
- [once for all](#once-for-all)

<!-- /TOC -->

## meta-learning

参考[https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)

## auto-ml

参考[https://www.automl.org/wp-content/uploads/2018/12/AutoML-Tutorial-NeurIPS2018-MetaLearning.pdf](https://www.automl.org/wp-content/uploads/2018/12/AutoML-Tutorial-NeurIPS2018-MetaLearning.pdf)

参考[AutoML研究综述：让AI学习设计AI](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650761726&idx=4&sn=0ce08475039d9c890a6ab8fe5bf85d6f&chksm=871aad80b06d2496af89ecd96c986dbcf419c4083032c6b6bc85a692f96a5996fe19025b3c2b&mpshare=1&scene=1&srcid=&pass_ticket=csFmp%2BqPqpbOEtBCr9byDm0vHyp83ccxf21EyZaHyV%2BoFQOLINXIlgzuTkVvCg24#rd)

参考[专栏 \| 神经网络架构搜索（NAS）综述（附AutoML资料推荐）](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650747841&idx=5&sn=5391948dfdd125be21f7cdd12a6318d1&chksm=871af7bfb06d7ea9e6b1e98aad6a9d1b1083e63ba940decee7e9290c807c73ecefe84e5c7694&scene=21#wechat_redirect)

参考[KDD Cup 2019 AutoML Track冠军深兰科技DeepBlueAI团队技术分享 \| 开源代码](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247498834&idx=1&sn=f16cb40771137d0c609bc420b556ee69&chksm=96ea23d2a19daac49f3cb84e44a7198f779a77daa84dbb71d9e93aa8b8949f874565da9f4c72&mpshare=1&scene=1&srcid=&sharer_sharetime=1565158428082&sharer_shareid=8e95986c8c4779e3cdf4e60b3c7aa752&pass_ticket=Kz97uXi0CH4ceADUC3ocCNkjZjy%2B0DTtVYOM7n%2FmWttTt5YKTC2DQT9lqCel7dDR#rd)

参考[AutoML: A Survey of the State-of-the-Art](https://arxiv.org/pdf/1908.00709v1)

在特定领域构建高质量的深度学习系统不仅耗时，而且需要大量的资源和人类的专业知识。为了缓解这个问题，许多研究正转向自动机器学习。本文是一个全面的 AutoML 论文综述文章，介绍了最新的 SOTA 成果。首先，文章根据机器学习构建管道的流程，介绍了相应的自动机器学习技术。然后总结了现有的神经架构搜索（NAS）研究。论文作者同时对比了 NAS 算法生成的模型和人工构建的模型。最后，论文作者介绍了几个未来研究中的开放问题。

### fairnas

参考[超越MnasNet、Proxyless：小米开源全新神经架构搜索算法FairNAS](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650765320&idx=3&sn=76f3d759c7a970950a476ad938816b66&chksm=871abc76b06d3560c4ad5b3978aa9338f067fdaf22eb763a6269780878c1227ed22085c7edd4&scene=0&xtrack=1&pass_ticket=g0RhlU91yTm4YwdL6HxxS6fDU%2FNvWsf8uqd5BGk9%2Fewn4u2UU5gMclDp6uVTk%2Bm3#rd)

[FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search](https://arxiv.org/pdf/1907.01845.pdf)

源码：[https://github.com/fairnas/FairNAS](https://github.com/fairnas/FairNAS)

### efficientnet

参考[https://daiwk.github.io/posts/cv-efficientnet.html](https://daiwk.github.io/posts/cv-efficientnet.html)

### adanet

参考[https://daiwk.github.io/posts/platform-adanet.html](https://daiwk.github.io/posts/platform-adanet.html)

### mdenas

参考[ICCV 2019 \| 四小时搜索NN结构，厦大提出快速NAS检索方法](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650768900&idx=4&sn=d61886200c847f7beca7b68ad1416539&chksm=871a427ab06dcb6c91d4d579cd9e7a95fc3dd08d93110f7ab495fb31d02fc693421869888975&scene=0&xtrack=1&pass_ticket=mmBhl6hER5JU9q0KMKTTFnbwPDksdn18kk%2FlW9Ih3p2TCzi4%2BlfisKHhCysHq%2Bou#rd)

[Multinomial Distribution Learning for Effective Neural Architecture Search](https://arxiv.org/abs/1905.07529)

[https://github.com/tanglang96/MDENAS](https://github.com/tanglang96/MDENAS)

近年来，通过神经架构搜索（NAS）算法生成的架构在各种计算机视觉任务中获得了极强的的性能。然而，现有的 NAS 算法需要再上百个 GPU 上运行 30 多天。在本文中，我们提出了一种基于多项式分布估计快速 NAS 算法，它将搜索空间视为一个多项式分布，我们可以通过采样-分布估计来优化该分布，从而将 NAS 可以转换为分布估计/学习。

除此之外，本文还提出并证明了一种保序精度排序假设，进一步加速学习过程。在 CIFAR-10 上，通过我们的方法搜索的结构实现了 2.55％的测试误差，GTX1080Ti 上仅 4 个 GPU 小时。在 ImageNet 上，我们实现了 75.2％的 top1 准确度。

### ProxylessNAS & HAQ

[寻找最佳的神经网络架构，韩松组两篇论文解读](https://mp.weixin.qq.com/s/ulrPhfsPunKAWYohBhkh9w)

[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/pdf/1812.00332v2.pdf),  ICLR2019

github: [https://github.com/MIT-HAN-LAB/ProxylessNAS](https://github.com/MIT-HAN-LAB/ProxylessNAS)

[HAQ: Hardware-Aware Automated Quantization with Mixed Precision](https://arxiv.org/pdf/1811.08886.pdf), CVPR2019

### 进化论方法

[亚马逊：自动选择AI模型，进化论方法效率更高！](https://mp.weixin.qq.com/s/xVhaIEuWUgPP8Va-hkYjFg)

来自亚马逊的研究人员探索出了一种可适用于任何计算模型的技术，条件是该模型可以计算出与图灵机相同的功能。（这里的“图灵机”是指定义抽象机的模型，可以根据规则来操纵符号。）

“无论使用哪种学习算法，选择哪种体系结构或调整训练参数（例如批规模或学习率），选择神经体系结构都不可能为给定的机器学习问题提供最佳解决方案，”Alexa AI机器学习平台服务组织的研究工程师，论文的主要作者温特表示。“只有考虑到尽可能多的可能性，才能确定一种在理论上保证计算准确性的体系结构。”

为此，研究团队评估了函数逼近问题的解决方案，这是AI算法搜索参数以逼近目标函数输出的方式的数学抽象方法。研究人员将其重新制定为发现一个估计目标函数输出的已知函数序列的问题，以获取更大的系统建模优势。

研究人员的研究表明，应该选择AI模型的组成部分，以确保它们具有“图灵等效性”。研究人员认为，最好通过自动搜索来识别模型，使用程序来设计特定任务的AI模型架构。这种搜索中的算法会首先生成用于解决问题的其他候选算法，然后将性能最佳的候选者彼此组合并再次进行测试。
 
“本文中……可立即应用的结果是鉴定遗传算法，更具体地说，是协同进化算法，其性能指标取决于彼此之间的相互作用，这是寻找最佳（或接近最佳）架构的最实用方法，”论文作者写道。“基于经验，许多研究人员得出的结论是，协同进化算法提供了构建机器学习系统的最佳方法。但是本文中的函数逼近框架有助于为他们的直觉提供更安全的理论基础。”
 
亚马逊并不是唯一一个倡导采用进化方法进行AI架构搜索的机构。今年7月，Uber为名为EvoGrad的进化算法开源了开发资源库。去年10月，Google推出了AdaNet，这是一种用于组合机器学习算法以获得更好的预测观点的工具。

[On the Bounds of Function Approximations](https://arxiv.org/abs/1908.09942)

### autogan

[[https://daiwk.github.io/posts/cv-autogan.html](https://daiwk.github.io/posts/cv-autogan.html]([https://daiwk.github.io/posts/cv-autogan.html](https://daiwk.github.io/posts/cv-autogan.html)

## once for all

参考[韩松等人提出NN设计新思路：训练一次，全平台应用](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650769015&idx=5&sn=8e490488bdb2cd6a0aa28fa89083d55f&chksm=871a4209b06dcb1f1ec06b3a74bffad1eade7f69e790516bfa9f7c332fe22bddb83df90d9e41&scene=0&xtrack=1&pass_ticket=mmBhl6hER5JU9q0KMKTTFnbwPDksdn18kk%2FlW9Ih3p2TCzi4%2BlfisKHhCysHq%2Bou#rd)

[Once for All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/pdf/1908.09791.pdf)

如要有效地部署深度学习模型，需要专门的神经网络架构，以便最好地适应不同的硬件平台和效率限制条件（定义为部署场景（deployment scenario））。传统的方法要么是人工设计，要么就是使用 AutoML（自动机器学习）来搜索特定的神经网络，再针对每个案例从头开始训练。这些方法的成本高昂，而且难以扩展，因为它们的训练成本与部署场景的数量有关。

本研究为高效神经网络设计引入了一种 Once for All（OFA/一劳永逸）方法，可处理很多部署场景。这种新方法的特点是分离了模型训练与架构搜索过程。这种方法不会针对每种案例都训练一个专用模型，而是训练一个支持多种不同架构设置（深度、宽度、核大小和分辨率）的 OFA 网络。

然后给定一个部署场景，再通过选择 OFA 网络中特定的子网络来搜索合适的结果，这个过程无需训练。因此，专用模型的训练成本就从 O(N) 降到了 O(1)。但是，我们却难以防止许多不同子网络之间的干扰。针对这一问题，MIT 的这些研究者提出了渐进式收束算法（progressive shrinking algorithm）。该算法能够训练支持超过 10^19 个子网络的 OFA 网络，同时还能保持与独立训练的网络一样的准确度，从而节省非重复性工程开发（non-recurring engineering/NRE）成本。

研究者在多种不同的硬件平台（移动平台/CPU/GPU）和效率限制条件上进行了广泛的实验，结果表明：相比于当前最佳（SOTA）的神经架构搜索（NAS）方法，OFA 能稳定地取得同等水平（或更好）的 ImageNet 准确度。值得注意的是，OFA 在处理多部署场景（N）时的速度比 NAS 快几个数量级。当 N=40 时，OFA 所需的 GPU 工作小时数比 ProxylessNAS 少 14 倍、比 FBNet 少 16 倍、比 MnasNet 少 1142 倍。部署场景越多，则相比于 NAS 就节省越多。

