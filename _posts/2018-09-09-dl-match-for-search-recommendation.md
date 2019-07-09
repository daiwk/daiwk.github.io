---
layout: post
category: "dl"
title: "match for search recommendation（深度学习部分）"
tags: [match, search, recommend, lihang, sigir18, www18 ]
---

<!-- TOC -->

- [概述](#%E6%A6%82%E8%BF%B0)
- [搜索领域的deep match](#%E6%90%9C%E7%B4%A2%E9%A2%86%E5%9F%9F%E7%9A%84deep-match)
  - [学习搜索representation](#%E5%AD%A6%E4%B9%A0%E6%90%9C%E7%B4%A2representation)
  - [学习搜索match函数](#%E5%AD%A6%E4%B9%A0%E6%90%9C%E7%B4%A2match%E5%87%BD%E6%95%B0)
    - [学习query-document的matching matrix](#%E5%AD%A6%E4%B9%A0query-document%E7%9A%84matching-matrix)
      - [ARC-II](#ARC-II)
      - [MatchPyramid](#MatchPyramid)
      - [Match-SRNN](#Match-SRNN)
      - [K-NRM](#K-NRM)
      - [Conv-KNRM](#Conv-KNRM)
    - [使用attention model进行match](#%E4%BD%BF%E7%94%A8attention-model%E8%BF%9B%E8%A1%8Cmatch)
- [推荐的deep match](#%E6%8E%A8%E8%8D%90%E7%9A%84deep-match)
  - [学习推荐representation](#%E5%AD%A6%E4%B9%A0%E6%8E%A8%E8%8D%90representation)
    - [Pure CF models](#Pure-CF-models)
      - [DeepMF](#DeepMF)
      - [AutoRec](#AutoRec)
      - [CDAE](#CDAE)
    - [CF with side information](#CF-with-side-information)
      - [DCF](#DCF)
      - [DUIF](#DUIF)
      - [ACF](#ACF)
      - [CKB](#CKB)
  - [学习推荐match函数](#%E5%AD%A6%E4%B9%A0%E6%8E%A8%E8%8D%90match%E5%87%BD%E6%95%B0)
    - [Pure CF models的match学习](#Pure-CF-models%E7%9A%84match%E5%AD%A6%E4%B9%A0)
      - [基于Neural Collaborative Filtering框架](#%E5%9F%BA%E4%BA%8ENeural-Collaborative-Filtering%E6%A1%86%E6%9E%B6)
        - [NeuMF](#NeuMF)
        - [NNCF](#NNCF)
        - [ConvNCF](#ConvNCF)
      - [基于Translation框架](#%E5%9F%BA%E4%BA%8ETranslation%E6%A1%86%E6%9E%B6)
        - [TransRec](#TransRec)
        - [LRML](#LRML)
    - [Feature-based models的match学习](#Feature-based-models%E7%9A%84match%E5%AD%A6%E4%B9%A0)
      - [基于MLP](#%E5%9F%BA%E4%BA%8EMLP)
        - [Wide&Deep](#WideDeep)
      - [基于FM](#%E5%9F%BA%E4%BA%8EFM)
        - [Neural FM](#Neural-FM)
        - [Attentional FM](#Attentional-FM)
      - [基于树](#%E5%9F%BA%E4%BA%8E%E6%A0%91)
        - [GB-CENT](#GB-CENT)
        - [DEF](#DEF)
        - [TEM](#TEM)

<!-- /TOC -->

有两个版本，一个是www18的：[https://www.comp.nus.edu.sg/~xiangnan/papers/www18-tutorial-deep-matching.pdf](https://www.comp.nus.edu.sg/~xiangnan/papers/www18-tutorial-deep-matching.pdf)

一个是sigir18的：[https://www.comp.nus.edu.sg/~xiangnan/sigir18-deep.pdf](https://www.comp.nus.edu.sg/~xiangnan/sigir18-deep.pdf)

好像都403了。。可以看这个[http://www.hangli-hl.com/uploads/3/4/4/6/34465961/wsdm_2019_tutorial.pdf](http://www.hangli-hl.com/uploads/3/4/4/6/34465961/wsdm_2019_tutorial.pdf)

sigir的这个比较新。。看之

这里讲深度学习部分，传统部分见：[https://daiwk.github.io/posts/dl-match-for-search-recommendation-traditional.html](https://daiwk.github.io/posts/dl-match-for-search-recommendation-traditional.html)

## 概述

## 搜索领域的deep match

### 学习搜索representation

### 学习搜索match函数

#### 学习query-document的matching matrix

##### ARC-II

AAAI’16 [Convolutional Neural Network Architectures for Matching Natural Language Sentences](http://www.hangli-hl.com/uploads/3/1/6/8/3168008/hu-etal-nips2014.pdf)

##### MatchPyramid

AAAI’16 [Text Matching as Image Recognition](https://arxiv.org/pdf/1602.06359.pdf)

##### Match-SRNN

IJCAI’16 [Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN](https://arxiv.org/pdf/1604.04378.pdf)

##### K-NRM

SIGIR'17 [End-to-End Neural Ad-hoc Ranking with Kernel Pooling](http://www.cs.cmu.edu/~zhuyund/papers/end-end-neural.pdf)

##### Conv-KNRM

WSDM'18 [Convolutional Neural Networks for So-Matching N-Grams in Ad-hoc Search](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf)

#### 使用attention model进行match

EMNLP 2016 [A Decomposable Attention Model for Natural Language Inference](https://aclweb.org/anthology/D16-1244)

## 推荐的deep match

### 学习推荐representation

#### Pure CF models

##### DeepMF

Deep Matrix Factorization

##### AutoRec

Autoencoders Meeting CF

##### CDAE

Collaborative Denoising Autoencoder

#### CF with side information

##### DCF

Deep Collaborative Filtering via Marginalized DAE

##### DUIF

Deep User-Image Feature

##### ACF

Attentive Collaborative Filtering

##### CKB

Collaborative Knowledge Base Embeddings

### 学习推荐match函数

#### Pure CF models的match学习

##### 基于Neural Collaborative Filtering框架

###### NeuMF

###### NNCF

###### ConvNCF

##### 基于Translation框架

###### TransRec

要求`\(Head + Relation \approx Tail\)`，也就是说，想让这两个向量尽量是同一个向量，那cos相似度就没啥用了，因为cos只能表示夹角尽量小，可能向量的长度会差很远，所以呢，可以用L1(曼哈顿距离)或者L2距离(欧几里得距离)！！！

###### LRML

直接用欧氏距离，relation向量是通过attention学到的

#### Feature-based models的match学习

##### 基于MLP

###### Wide&Deep

##### 基于FM

###### Neural FM

###### Attentional FM

##### 基于树

###### GB-CENT

###### DEF

###### TEM
