---
layout: post
category: "nlp"
title: "word2vec"
tags: [word2vec, ngram, nnlm, cbow, c-skip-gram, 统计语言模型]
---

目录

* [1. 统计语言模型](#1-统计语言模型)
  * [N-gram模型](#n-gram模型)
  * [神经网络语言模型（NNLM）](#神经网络语言模型nnlm)
* [2. CBOW](#2-cbow)
* [3. Continuous skip-gram](#3-continuous-skip-gram)

参考cdsn博客：[http://blog.csdn.net/zhaoxinfan/article/details/27352659](http://blog.csdn.net/zhaoxinfan/article/details/27352659)

参考paddlepaddle book: [https://github.com/PaddlePaddle/book/blob/develop/04.word2vec/README.cn.md](https://github.com/PaddlePaddle/book/blob/develop/04.word2vec/README.cn.md)

Word2vec的原理主要涉及到**统计语言模型**（包括N-gram模型和神经网络语言模型(nnlm)），**continuous bag-of-words**模型以及**continuous skip-gram**模型。

语言模型旨在为语句的联合概率函数`\(P(w_1,...,w_T)\)`建模。语言模型的目标是，希望模型对有意义的句子赋予大概率，对没意义的句子赋予小概率。 常用条件概率表示语言模型：

`\[
P(w_1, ..., w_T) = \prod_{t=1}^TP(w_t | w_1, ... , w_{t-1})
\]`

# 1. 统计语言模型
## N-gram模型

## 神经网络语言模型（NNLM）

# 2. CBOW

# 3. Continuous skip-gram

