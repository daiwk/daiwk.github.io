---
layout: post
category: "dl"
title: "match for search recommendation"
tags: [match, search, recommend, lihang, sigir18, www18 ]
---

<!-- TOC -->

- [aaa](#aaa)
- [xx](#xx)
    - [xxx](#xxx)
        - [TransRec](#transrec)
        - [Latent Relational Metric Learning](#latent-relational-metric-learning)

<!-- /TOC -->

有两个版本，一个是www18的：[https://www.comp.nus.edu.sg/~xiangnan/papers/www18-tutorial-deep-matching.pdf](https://www.comp.nus.edu.sg/~xiangnan/papers/www18-tutorial-deep-matching.pdf)

一个是sigir18的：[https://www.comp.nus.edu.sg/~xiangnan/sigir18-deep.pdf](https://www.comp.nus.edu.sg/~xiangnan/sigir18-deep.pdf)

好像都403了。。可以看这个[http://www.hangli-hl.com/uploads/3/4/4/6/34465961/wsdm_2019_tutorial.pdf](http://www.hangli-hl.com/uploads/3/4/4/6/34465961/wsdm_2019_tutorial.pdf)

sigir的这个比较新。。看之

## aaa

## xx

### xxx

#### TransRec

要求`\(Head + Relation \approx Tail\)`，也就是说，想让这两个向量尽量是同一个向量，那cos相似度就没啥用了，因为cos只能表示夹角尽量小，可能向量的长度会差很远，所以呢，可以用L1(曼哈顿距离)或者L2距离(欧几里得距离)！！！

#### Latent Relational Metric Learning

直接用欧氏距离，relation向量是通过attention学到的

