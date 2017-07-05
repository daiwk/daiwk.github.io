---
layout: post
category: "cv"
title: "图文相关性模型简介"
tags: [图文相关性, ]
---


## 1. 相关模型

### 1. basic模型

文本采用word2vec获取标题向量，cos_sim计算 图文的相关性，然后用pairwise训练

<html>
<br/>

<img src='../assets/img_txt_sim_basic.svg' style='max-height: 400px'/>
<br/>

</html>

### 2. 升级文本表示为bi-lstm

目前简单抽取1k的图文配对（1k正+1k随机产出的负例），
+ 如果正例的相关性>负例的相关性，则暂认为：判定有效
+ 模型误判：如果正例判定不相关（<0）或负例判定相关(>0)

<html>
<br/>

<img src='../assets/img_txt_sim_lstm.svg' style='max-height: 400px'/>
<br/>

</html>

