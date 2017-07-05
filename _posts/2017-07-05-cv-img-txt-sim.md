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

<html>
<br/>

<img src='../assets/img_txt_sim_lstm.svg' style='max-height: 400px'/>
<br/>

</html>

