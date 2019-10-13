---
layout: post
category: "nlp"
title: "文本生成模型实践"
tags: [文本生成模型, transsent, ]
---

目录

<!-- TOC -->

- [keras+lstm](#keraslstm)
- [transsent](#transsent)

<!-- /TOC -->

## keras+lstm

[用自己的风格教AI说话，语言生成模型可以这样学](https://mp.weixin.qq.com/s/eoPFC6fms1qhGTOqCk-3Qg)

[https://github.com/maelfabien/Machine_Learning_Tutorials](https://github.com/maelfabien/Machine_Learning_Tutorials)

[https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms)

## transsent

[节后收心困难？这15篇论文，让你迅速找回学习状态](https://mp.weixin.qq.com/s/aaz-s87vorroyepNCd9-AA)

[TransSent: Towards Generation of Structured Sentences with Discourse Marker](https://arxiv.org/abs/1909.05364)

[https://github.com/1024er/TransSent_dataset](https://github.com/1024er/TransSent_dataset)

本文来自中科院，论文将知识图谱的 Trans 思想引入到句子表示空间，提出了一个新的任务 Sentence Transfer，为了解决这个任务构建了三个数据集。人会在写作的时候使用连词（discourse marker，比如 and, but, if...）来衔接子句（discourses），连词用来显式地表达前后子句之间的语义关系。

因此作者认为在 embedding space 中，子句之间可以利用连词（表示的关系）进行迁移/翻译。在学习出句子之间的 Trans 模型后，可以用来分阶段生成结构化长句：首先生成首子句，然后选择一种关系，再 Trans 生成尾子句。这样的生成方式可以和现有的问答等任务结合。作者还给出了一个数据集上的实验结果。总体来说本文给出了一个新的生成的思路，值得发掘。
