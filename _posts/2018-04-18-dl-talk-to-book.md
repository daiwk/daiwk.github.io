---
layout: post
category: "dl"
title: "talk to book"
tags: [talk to book, Universal Sentence Encoder]
---

目录

<!-- TOC -->

- [实验效果：](#实验效果)
    - [准确率](#准确率)
    - [性能](#性能)
        - [计算复杂度](#计算复杂度)
        - [内存占用](#内存占用)

<!-- /TOC -->

参考：

[人人都可参与的AI技术体验：谷歌发布全新搜索引擎Talk to Books](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650740857&idx=1&sn=be95f7c9f04c1cd8b4a8b4e5b6b5de5b&chksm=871adc07b06d55118fc6d66522a2e7e3e6fff3231775b050842a70b9df1326ae9e9a128c4702&mpshare=1&scene=1&srcid=0414zG2CmGMaxS6Oo7jDtihI&pass_ticket=HoHoizuZ8hXEv%2BpUKGlCMf6B4i260rN3vajF9BgQ0sjDtoVg7DAO2SlGSICkvRb1#rd)

[Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)

pre-trained model:
[https://tfhub.dev/google/universal-sentence-encoder/1](https://tfhub.dev/google/universal-sentence-encoder/1)

使用示例：
[https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/1](https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/1)

my notebook: [https://github.com/daiwk/workspace_tf/blob/master/talk_to_books/Semantic_Similarity_with_TF_Hub_Universal_Encoder.ipynb](https://github.com/daiwk/workspace_tf/blob/master/talk_to_books/Semantic_Similarity_with_TF_Hub_Universal_Encoder.ipynb)

项目链接：[https://research.google.com/semanticexperiences/](https://research.google.com/semanticexperiences/)

主要的两篇参考文献：

+ attention is all you need，解读可以参考[https://daiwk.github.io/posts/platform-tensor-to-tensor.html](https://daiwk.github.io/posts/platform-tensor-to-tensor.html)
+ DAN，即[Deep Unordered Composition Rivals Syntactic Methods for Text Classification](https://www.cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf)

## 实验效果：

### 准确率

在transfer learning的任务SST上，USE_T（Universal Sentence Encoder + Transformer）只需要1k的训练数据（这个数据集总共有67.3k的训练数据），就能够达到其他很多模型使用全量训练数据得到的准确率。

### 性能

假设句子长度为n。

#### 计算复杂度

+ Transformer: `\(O(n^2)\)`
+ DAN: `\(O(n)\)`

#### 内存占用

+ Transformer: `\(O(n^2)\)`。但对于短句，因为Transformer只要存储unigram的embedding，所以占用的内存几乎是DAN的一半。
+ DAN: 与句子长度无关，由用来存储unigram和bigram的embedding的参数决定
