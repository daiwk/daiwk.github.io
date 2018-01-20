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

## Research Metrics

## Courses, Tutorials and Books

# Engineering

## Higher-order gradients

## Distributed PyTorch

## Closer to NumPy

## Sparse Tensors

## Performance

## Exporting models to production — ONNX Support and the JIT compiler

