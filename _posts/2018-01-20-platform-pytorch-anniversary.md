---
layout: post
category: "platform"
title: "pytorch一周年"
tags: [pytorch, ]
---

目录

<!-- TOC -->

- [Community](#community)
    - [Research papers, packages and Github](#research-papers-packages-and-github)
        - [cycle-GAN](#cycle-gan)
        - [opennmt](#opennmt)
        - [超分辨率](#超分辨率)
        - [PyTorch-QRNN](#pytorch-qrnn)
        - [Pyro & ProbTorch](#pyro--probtorch)
        - [pix2pixHD, sentiment neuron & flownet2](#pix2pixhd-sentiment-neuron--flownet2)
        - [allenNLP](#allennlp)
        - [DSB2017冠军](#dsb2017冠军)
        - [可视化](#可视化)
        - [Facebook AI Research](#facebook-ai-research)
    - [Metrics](#metrics)
    - [Research Metrics](#research-metrics)
    - [Courses, Tutorials and Books](#courses-tutorials-and-books)
- [Engineering](#engineering)
    - [Higher-order gradients](#higher-order-gradients)
    - [Distributed PyTorch](#distributed-pytorch)
    - [Closer to NumPy](#closer-to-numpy)
    - [Sparse Tensors](#sparse-tensors)
    - [Performance](#performance)
        - [Reducing framework overhead by 10x across board](#reducing-framework-overhead-by-10x-across-board)
        - [ATen](#aten)
    - [Exporting models to production — ONNX Support and the JIT compiler](#exporting-models-to-production--onnx-support-and-the-jit-compiler)

<!-- /TOC -->

参考 [PyTorch一周年战绩总结：是否比TensorFlow来势凶猛？](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650736406&idx=1&sn=db8da5ddc9a9cf86e804d29eb817f078&chksm=871ac368b06d4a7ec4854a44cc758a2d9610a350569be43209d4f99868e7341e7a551155079e&mpshare=1&scene=1&srcid=0120WbpeJXZUHZpvcIhqA8Xu&pass_ticket=xK%2FffWIobjEWlqRcODdvVXVcND5Es%2FtthdouMWebLV70o%2F654l0vh1big6xAGCGm#rd)

原文：[http://pytorch.org/2018/01/19/a-year-in.html](http://pytorch.org/2018/01/19/a-year-in.html)


# Community

## Research papers, packages and Github

人们一起创建了 **torchtext**、**torchvision** 和 **torchaudio**，以便利化平民化不同领域的研究。

首个 PyTorch 社区工具包（被命名为 Block）来自 Brandon Amo，有助于更轻松地处理块矩阵（block matrix）。来自 CMU 的 Locus 实验室后来继续公布 PyTorch 工具包及其大部分研究的实现。首个研究论文代码来自 Sergey Zagoruyko，论文名称为《Paying more attention to attention》。


### cycle-GAN

来自 U.C.Berkeley 的 Jun-Yan Zhu、Taesung Park、Phillip Isola、Alyosha Efros 及团队发布了非常流行的 **Cycle-GAN** 和 **pix2pix**，用于图像转换。

torch版：[https://github.com/junyanz/CycleGAN](https://github.com/junyanz/CycleGAN)

pytorch版：[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

论文链接：[https://arxiv.org/abs/1703.10593](https://arxiv.org/abs/1703.10593)

介绍：[让莫奈画作变成照片：伯克利图像到图像翻译新研究](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650725257&idx=4&sn=bf367ff90e03f8189f7c67ae0e5ab76f&chksm=871b1ff7b06c96e1e355d8b360abd0c256af04e2ba72a8d2a3364bfea8ff80b347a734d17e9d&scene=21#wechat_redirect)

### opennmt

[哈佛大学NLP组开源神经机器翻译工具包OpenNMT：已达到生产可用水平](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650721602&idx=3&sn=4f80dae1dfe49c1151288a60731d4b40&chksm=871b093cb06c802abc2c4f3dc7c2989caad155308667da292888a26c6297be28c37018e40769&scene=21#wechat_redirect)

HarvardNLP 和 Systran 的研究者开始使用 PyTorch 开发和提升 OpenNMT，它最初开始于 Facebook Adam Lerer 的 [Lua]Torch 代码最初的再实现。

项目主页：[http://opennmt.net/](http://opennmt.net/)

pytorch版：[https://github.com/OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)

tf版：
[https://github.com/OpenNMT/OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf)

### 超分辨率

来自 Twitter 的 MagicPony 团队贡献了其超分辨率研究示例的 PyTorch 实现【直接集成进pytorch的examples中】[https://github.com/pytorch/examples/tree/master/super_resolution](https://github.com/pytorch/examples/tree/master/super_resolution)

### PyTorch-QRNN

Salesforce 发布了若干个工具包，包括其亮点成果 PyTorch-QRNN，这是一种新型 RNN，相比于 CuDNN 优化的标准 LSTM 可提速 2 到 17 倍。James Bradbury 及其团队是 PyTorch 社区中最活跃和最有吸引力的团队之一。

pytorch：[https://github.com/salesforce/pytorch-qrnn](https://github.com/salesforce/pytorch-qrnn)

### Pyro & ProbTorch

来自 Uber、Northeaster、Stanford 的研究者围绕着其工具包 Pyro 和 ProbTorch，形成了一个活跃的概率编程社区。他们正在积极开发 torch.distributions 核心工具包。该社区非常活跃，快速发展，我们联合 Fritz Obermeyer、Noah Goodman、Jan-Willem van de Meent、Brooks Paige、Dustin Tran 及其他 22 名参会者在 NIPS 2017 上举办了首次 PyTorch 概率编程会议，共同探讨如何使世界贝叶斯化。

pyro：

[http://pyro.ai/](http://pyro.ai/)

参考：[Uber 与斯坦福大学开源深度概率编程语言 Pyro：基于 PyTorch](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650732815&idx=1&sn=eddc61facb9f25da638be9fc494302e2&chksm=871b3d71b06cb46703d9781830dc6949ab14b531026a91b84366b9c5abd8b59c0f4154e1a115&scene=21#wechat_redirect)

probtorch：

[https://github.com/probtorch/probtorch](https://github.com/probtorch/probtorch)

### pix2pixHD, sentiment neuron & flownet2

英伟达研究者发布了三个高质量 repo，实现了 pix2pix-HD、Sentiment Neuron 和 FlowNet2。对 PyTorch 中不同数据并行模型的扩展性分析对整个社区都很有益。

pix2pix-HD：

[https://github.com/NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD)

sentiment neuron: 
[https://github.com/NVIDIA/sentiment-discovery](https://github.com/NVIDIA/sentiment-discovery)

flownet2: 
[https://github.com/NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)

### allenNLP

艾伦人工智能研究所发布 AllenNLP，包括多个 NLP 先进模型：标准 NLP 任务的参考实现和易用 web demo。

代码：[https://github.com/allenai/allennlp](https://github.com/allenai/allennlp)

allenNLP: [http://allennlp.org/](http://allennlp.org/)

demo[包括machine-comprehension/textuale-ntailment/semantic-role-labeling/coreference-resolution/named-entity-recognitio]：[http://demo.allennlp.org/machine-comprehension](http://demo.allennlp.org/machine-comprehension)

### DSB2017冠军

六月份，我们还首次取得了 Kaggle 竞赛冠军（团队 grt123）。他们获得了 2017 数据科学杯（关于肺癌检测）【DataScience Bowl 2017 on Lung Cancer detection】的冠军，后来公开了其 PyTorch 实现：

[https://github.com/lfz/DSB2017](https://github.com/lfz/DSB2017)

### 可视化

在可视化方面，Tzu-Wei Huang 实现了 TensorBoard-PyTorch 插件
[https://github.com/lanpa/tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch)

Facebook AI Research 发布了与 PyTorch 兼容的 **visdom** 可视化包。

[https://github.com/facebookresearch/visdom](https://github.com/facebookresearch/visdom)

### Facebook AI Research

Facebook AI Research 发布了多个项目，如 ParlAI、fairseq-py、VoiceLoop 和 FaderNetworks，在多个领域中实现了先进的模型和接口数据集。由于空间有限，这里就不将优秀项目一一列出，详细列表可参阅：[https://github.com/facebookresearch/](https://github.com/facebookresearch/)。

ParlAI:

[https://github.com/facebookresearch/ParlAI](https://github.com/facebookresearch/ParlAI)

fairseq-py:
[https://github.com/facebookresearch/fairseq-py](https://github.com/facebookresearch/fairseq-py)

VoiceLoop:

[https://github.com/facebookresearch/loop](https://github.com/facebookresearch/loop)

FaderNetworks: 

[https://github.com/facebookresearch/FaderNetworks](https://github.com/facebookresearch/FaderNetworks)


## Metrics

+ 在 Github 上有 87769 行代码引入 Torch。
+ 在 Github 上有 3983 个 repository 在名字或者描述中提到了 PyTorch。
+ PyTorch binary 下载量超过 50 万，具体数字是 651916。
+ 在论坛上，有 5400 名用户发表了 21500 条讨论，涉及 5200 个主题。
+ 自发布以来，在 Reddit 上的/r/machinelearning 主题中有 131 条讨论提到了 PyTorch。同期，TensorFlow 被提及的次数为 255。

pytorch v.s. tensorflow:
[PyTorch和TensorFlow到底哪个更好？看看一线开发者怎么说](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650723769&idx=1&sn=17565e650771699ceddabb214d485626&chksm=871b11c7b06c98d1c76623f7c90120e363cc43462b74f22c27038e324c2975ec4d0db5b483c1&scene=21#wechat_redirect)

[TensorFlow开源一周年：这可能是一份最完整的盘点](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650720407&idx=1&sn=768d7248e0ab5fa469dbae86d11152e1&chksm=871b0ce9b06c85ffefa2e0c8f6fb7ae4cc1c0500cda7bad008fe68ed6b87d7c8765d138e1fd1&scene=21#wechat_redirect)

## Research Metrics

PyTorch 是一个专注于研究的框架。所以与衡量它的指标包括 PyTorch 在机器学习研究论文中的使用。
+ 在 ICLR 2018 学术会议提交的论文中，有 87 篇提到了 PyTorch，相比之下 TensorFlow 228 篇，Keras 42 篇，Theano 和 Matlab 是 32 篇。
+ 按照月度来看，arXiv 论文提到 PyTorch 框架的有 72 篇，TensorFlow 是 273 篇，Keras 100 篇，Caffe 94 篇，Theano 53 篇。

## Courses, Tutorials and Books

Sasank Chilamkurthy 承担了改进教程的任务，教程详见：[http://pytorch.org/tutorials/](http://pytorch.org/tutorials/)

Sean Robertson 和 Justin Johnson 编写了 NLP 领域的全新教程，还有通过示例学习的教程。
+ [https://github.com/spro/practical-pytorch](https://github.com/spro/practical-pytorch)
+ [https://github.com/jcjohnson/pytorch-examples](https://github.com/jcjohnson/pytorch-examples)

Yunjey Choi 写了用 30 行或者更少的代码部署大多数模型的教程。每个新教程都帮助用户用不同的学习方法更快地找到适合自己的学习路径。
+ [https://github.com/yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)

Goku Mohandas 和 Delip Rao 把正在写的书中的代码做了改变，使用了 PyTorch。

我们看到，一些大学的机器学习课程是使用 PyTorch 作为主要工具讲授的，例如哈佛 CS 287。为了更进一步方便大众学习，我们还看到三个在线课程使用 PyTorch 讲授。

+ [https://harvard-ml-courses.github.io/cs287-web/](https://harvard-ml-courses.github.io/cs287-web/)

Fast.ai 的「Deep Learning for Coders」是个流行的在线课程。9 月份，Jeremy 和 Rachel 宣布下一个 fast.ai 的课程将几乎全部基于 PyTorch。

+ [http://www.fast.ai/2017/09/08/introducing-pytorch-for-fastai/](http://www.fast.ai/2017/09/08/introducing-pytorch-for-fastai/)

Ritchie Ng，在清华、新加坡国立大学都学习过的研究者，推出了名为「Practical Deep Learning with PyTorch」的 Udemy 课程。

+ [https://www.udemy.com/practical-deep-learning-with-pytorch/](https://www.udemy.com/practical-deep-learning-with-pytorch/)

来自香港科技大学的 Sung Kim 在 Yotube 上推出了面向普通观众的在线课程「PyTorch Zero to All」。

+ [https://www.youtube.com/playlist?list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m](https://www.youtube.com/playlist?list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m)
+ [四天速成！香港科技大学 PyTorch 课件分享](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650731685&idx=1&sn=9b8cfdf380ff9c8c91b45ebe7452f4ee&chksm=871b30dbb06cb9cd199412e72d7740970e82c7c61057473871287706a4239f3661eafbfd1630&scene=21#wechat_redirect)

# Engineering

## Higher-order gradients

随着多篇关于实现**梯度罚项**的论文的发表，以及**二阶梯度**法的不断研究发展，高阶梯度成为必需的热门功能。去年 8 月，我们实现了一个通用接口，**可使用 n 阶导数**，加快**支持高阶梯度函数的收敛**，截至写作本文时，几乎所有 ops 都支持此界面。

## Distributed PyTorch

去年 8 月，我们发布了一个小型分布式包，该包使用非常流行的 MPI 集合（MPI-collective）方法。**它有多个后端，如 TCP、MPI、Gloo 和 NCCL2**，以支持多种**CPU/GPU集合**操作和用例，这个包整合了 **Infiniband 和 RoCE** 等分布式技术。分布很难，我们在初始迭代时也有一些 bug。在后续版本中，我们作出了一些改进，使这个包更加稳定，性能也更强。

## Closer to NumPy

用户最大的一个需求是他们熟悉的 NumPy 功能。**Broadcasting** 和 **Advanced Indexing** 等功能方便、简洁，节约用户的时间。我们实现了这些功能，开始使我们的 API 更接近 NumPy。随着时间的进展，我们希望在合适的地方越来越接近 NumPy 的 API。

## Sparse Tensors

In March, we released a small package supporting sparse Tensors and in May we released CUDA support for the sparse package. The package is small and limited in functionality, and is used for **implementing Sparse Embeddings and commonly used sparse paradigms in deep learning**. This package is still small in scope and there’s demand to expand it — if you are interested in working on expanding the sparse package, reach out to us on our [Discussion Boards](https://discuss.pytorch.org/)

## Performance

性能是一场仍在进行中的战斗，尤其对于想要最大化灵活性的动态框架 PyTorch 而言。去年，从核心 Tensor 库到神经网络算子，我们改善了 PyTorch 在 board 上的性能，能在 board 上更快的编写微优化。

+ 我们添加了专门的 **AVX 和 AVX2 内部函数，用于 Tensor 运算**；
+ 写**更快的 GPU kernel**，用于常用的工作负载，如级联和 Softmax；
+ 为多个神经网络算子重写代码，如 nn.Embedding 和组卷积。

### Reducing framework overhead by 10x across board

由于 PyTorch 是动态图框架，我们在训练循环的**每次迭代时都要创建一个新图**。因此，框架开销必须很低，或者工作负载必须足够大来隐藏框架开销。去年 8 月，DyNet 的作者（Graham Neubig 及其团队）展示了 DyNet 在一些小型 NLP 模型上的速度快于 PyTorch。这是很有意思的一个挑战，我们开始重写 PyTorch 内部构件，**将框架开销从 10 微秒／算子降低到 1 微秒**。

### ATen

重新设计 PyTorch 内部构件的同时，我们也构建了 ATen C++11 库，该库现在主导 PyTorch 所有后端。ATen 具备一个类似 PyTorch Python API 的 API，使之成为**便于 Tensor 计算的 C++库**。ATen 可由 PyTorch 独立构建和使用。

[https://github.com/pytorch/pytorch/tree/master/aten](https://github.com/pytorch/pytorch/tree/master/aten)

## Exporting models to production — ONNX Support and the JIT compiler

我们收到的一个普遍请求是将 PyTorch 模型输出到另一个框架。**用户使用 PyTorch 进行快速研究，模型完成后，他们想将模型搭载到更大的项目中，而该项目只要求使用 C++。**

因此我们构建了 [**tracer**](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/tracer.h)，可将 PyTorch 模型输出为中间表示。用户可使用后续的 tracer 更高效地运行当前的 PyTorch 模型，或将其转换成 ONNX 格式以输出至 Caffe2、MXNet、TensorFlow 等其他框架，或直接搭载至硬件加速库，如 CoreML 或 TensorRT。今年，我们将更多地利用 JIT 编译器提升性能。
