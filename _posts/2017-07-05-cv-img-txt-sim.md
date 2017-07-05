---
layout: post
category: "cv"
title: "图文相关性模型简介"
tags: [图文相关性, ]
---


## 1. 相关模型

使用rank_cost([paddle_v2的layers](http://www.paddlepaddle.org/doc/api/v2/config/layer.html)):

`\[
\begin{align}\begin{aligned}C_{i,j} & = -\tilde{P_{ij}} * o_{i,j} + log(1 + e^{o_{i,j}})\\o_{i,j} & =  o_i - o_j\\\tilde{P_{i,j}} & = \{0, 0.5, 1\} \ or \ \{0, 1\}\end{aligned}\end{align}
\]`

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


### 2. 训练数据更新

之前都用title，现在改成单元内的bidword,pic的pair对。

<html>
<br/>

<img src='../assets/img_txt_sim_bidword.svg' style='max-height: 400px'/>
<br/>

</html>