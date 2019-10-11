---
layout: post
category: "nlp"
title: "flowseq"
tags: [flowseq, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

参考[节后收心困难？这15篇论文，让你迅速找回学习状态](https://mp.weixin.qq.com/s/aaz-s87vorroyepNCd9-AA)

[FlowSeq: Non-Autoregressive Conditional Sequence Generation with Generative Flow](https://arxiv.org/abs/1909.02480)

本文是 CMU 和 Facebook AI 联合发表于 EMNLP 2019 的工作。为了解决自自回归模型（auto regressive）在 Seq2Seq 问题上解码速度慢，只能利用一侧上下文信息等问题，提出了利用 generative flow 的非自回归模型（non-autoregressive）FlowSeq。在机器翻译任务上面的相比于之前的非自回归模型有显著提高，大大缩小了与自回归模型的差距。同时解码速度比自回归模型明显加快。

[https://github.com/XuezheMax/flowseq](https://github.com/XuezheMax/flowseq)
