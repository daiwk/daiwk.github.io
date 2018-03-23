---
layout: post
category: "platform"
title: "各框架的embedding实现"
tags: [embedding,]
---

目录

<!-- TOC -->

- [基本思想](#基本思想)
- [keras](#keras)
- [paddle](#paddle)
- [tensorflow](#tensorflow)
- [应用](#应用)
    - [xx率预估](#xx率预估)
        - [模型结构](#模型结构)
        - [编码方式（embedding）](#编码方式embedding)
            - [随机编码](#随机编码)
            - [挂靠编码](#挂靠编码)
            - [分词编码](#分词编码)
        - [Online Update](#online-update)

<!-- /TOC -->

## 基本思想

假设输入一个词，想把它embedding成一个向量。词典大小假设是`\(V\)`，embedding向量的维度假设是`\(K\)`。那么，相当于输入一个`\(V\)`维的**one-hot**向量，输出一个`\(K\)`维的向量，中间就是一个`\(V\times K\)`的矩阵`\(W\)`。假设`\(K=3,V=2\)`：

`\[
(x_0,x_1)\times \begin{bmatrix}
w_{00} & w_{01} &w_{02} \\ 
w_{10} & w_{11} & w_{12}
\end{bmatrix}=\begin{bmatrix}
x_0w_{00}+x_1w_{10}\\ 
x_0w_{01}+x_1w_{11}\\ 
x_0w_{02}+x_1w_{12}
\end{bmatrix}
\]`

举例，如果x=(0,1)那相当于是

`\[
(0,1)\times \begin{bmatrix}
w_{00} & w_{01} &w_{02} \\ 
w_{10} & w_{11} & w_{12}
\end{bmatrix}=\begin{bmatrix}
0w_{00}+1w_{10}\\ 
0w_{01}+1w_{11}\\ 
0w_{02}+1w_{12}
\end{bmatrix}=\begin{bmatrix}
w_{10}\\ 
w_{11}\\ 
w_{12}
\end{bmatrix}    
\]`

也就是说，**选取了权重矩阵的第1行的值**出来，**再转置一下，变成列向量**（即常说的lookup，或者paddle里的table_projection）。同理，如果x=(1,0)，那么相当于选取权重矩阵的第0行出来。

而特殊地，如果x不是one-hot，是一个可以有多个1的vector（例如，某个特征，可能同时取多个值，比如x有6维，每一维表示1种颜色，而这个特征是图中包含了哪几种颜色，可能有的图片里有3种颜色），那么，得到的每一维embedding就相当于把对应的几行选择出来，然后每行里每列对应的值相加。更直观地，假设`\(V=3,K=4\)`，假设x=(0,1,1)：

`\[
(0,1,1)\times \begin{bmatrix}
w_{00} & w_{01} &w_{02} &w_{03}\\ 
w_{10} & w_{11} & w_{12} &w_{13}\\
w_{20} & w_{21} & w_{22} &w_{23}\\
\end{bmatrix}=\begin{bmatrix}
0w_{00}+1w_{10}+1w_{20}\\ 
0w_{01}+1w_{11}+1w_{21}\\ 
0w_{02}+1w_{12}+1w_{22}\\
0w_{03}+1w_{13}+1w_{23}\\
\end{bmatrix}=\begin{bmatrix}
w_{10}+w_{20}\\ 
w_{11}+w_{21}\\ 
w_{12}+w_{22}\\
w_{13}+w_{23}
\end{bmatrix}
\]`

**相当于把权重矩阵的第1行和第2行拿出来，然后对应元素相加，再转置一下。**


## keras

中文文档：[https://keras.io/zh/layers/embeddings/](https://keras.io/zh/layers/embeddings/)

[https://keras.io/layers/embeddings/](https://keras.io/layers/embeddings/) keras的embedding是参考[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf)这个实现的，类似对输入数据进行dropout。

## paddle

[https://github.com/PaddlePaddle/Paddle/issues/2910](https://github.com/PaddlePaddle/Paddle/issues/2910)

## tensorflow

tensorflow的embedding操作：
[https://memeda.github.io/%E6%8A%80%E6%9C%AF/2017/04/13/tfembedding.html](https://memeda.github.io/%E6%8A%80%E6%9C%AF/2017/04/13/tfembedding.html)

如何使用预训练的embedding来初始化tf的embedding_lookup: 
[https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow](https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow)


## 应用

### xx率预估

[DNN在搜索场景中的应用](http://www.cnblogs.com/hujiapeng/p/6236857.html)

#### 模型结构

+ wide model
    + id feature: item_id, seller_id，学习已经出现过的商品，卖家在训练数据上的表现。
    + id cross feature: user_id x item_id , user_id x seller_id

连续值统计特征是非常有用的特征，Google的模型是把embedding向量和统计特征放到同一个DNN网络中学习，但实验发现这样会削弱统计特征的作用。我们为统计特征专门又组建了一个包含2个隐层的网路，并且为了增强非线性效果，激活函数从RELU改为TanH/Sigmiod。

+ deep model
    + 首先需要把离散特征（item_id，item_tag, user_id，user_tag，query_tag）embedding成连续特征。
    + 将embedding后的向量作为DNN的输入。考虑到最终线上预测性能的问题，目前我们的DNN网络还比较简单，只有1到2个隐层。

整体模型使用三层全连接层用于sparse+dense特征表征学习，再用两层全连接层用于点击/购买与否分类的统一深度学习模型解决方案：

+ 第一层为编码层，包含商品编码，店家编码，类目编码，品牌编码，搜索词编码和用户编码。 
+ 从第二层到第四层组成了“域间独立”的“行为编码网络”，其中第二层为针对稀疏编码特别优化过的全连接层( Sparse Inner Product Layer )，通过该层将压缩后的编码信息投影到16维的低维向量空间中，第三层和第四层均为普通全连接层，其输出维度分别为16和32。“行为编码网络”也可以被看做是针对域信息的二次编码，但是与第一层不同，这部分的最终输出是基于行为数据所训练出来的结果，具有行为上相似的商品或者用户的最终编码更相近的特性。
+ 第五层为concat层，其作用是将不同域的信息拼接到一起。
+ 第六层到第八层网络被称为“预测网络”，该部分由三层全连接组成，隐层输出分别为64,64和1。该部分的作用在于综合考虑不同域之间的信息后给出一个最终的排序分数。
+ 最后，Softmax作为损失函数被用在训练过程中; 非线性响应函数被用在每一个全连接之后。

在普适的CTR场景中，用户、商品、查询等若干个域的特征维度合计高达几十亿，假设在输入层后直接连接100个输出神经元的全连接层，那么这个模型的参数规模将达到千亿规模。直接接入全连接层将导致以下几个问题：1. 各个域都存在冷门的特征，这些冷门的特征将会被热门的特征淹没，基本不起作用，跟全连接层的连接边权值会趋向于0，冷门的商品只会更冷门。2. 模型的大小将会非常庞大，超过百G，在训练以及预测中都会出现很多工程上的问题。为了解决上述两个问题，本文引入了紫色编码层，具体分为以下两种编码方式：1. 随机编码 2. 挂靠编码，下面将对以上两种编码方式进行详细的描述。

#### 编码方式（embedding）
##### 随机编码

假设某一域的输入ID类特征的one-hot形式最大维度为N，

##### 挂靠编码

##### 分词编码

#### Online Update

双11当天数据分布会发生巨大变化，为了能更好的fit实时数据，我们将WDL的一部分参数做了在线实时训练。embeding层由于参数过多，并没有在线训练，其他模型参数都会在线学习更新。

deep端网络参数和wide端参数更新的策略有所不同，wide端是大规模稀疏特征，为了使训练结果有稀疏性，最好用FTRL来做更新。deep端都是稠密连续特征，使用的普通的SGD来做更新，学习率最好设置小一点。


和离线Batch training不同，Online learning会遇到一些特有的问题：

+ 实时streaming样本分布不均匀

现象：线上环境比较复杂，不同来源的日志qps和延迟都不同，造成不同时间段样本分布不一样，甚至在短时间段内样本分布异常。比如整体一天下来正负例1:9，如果某类日志延迟了，短时间可能全是负例，或者全是正例。 解决：Pairwise sampling。Pv日志到了后不立即产出负样本，而是等点击到了后找到关联的pv，然后把正负样本一起产出，这样的话就能保证正负样本总是1:9

+ 异步SGD更新造成模型不稳定

现象：权重学飘掉(非常大或者非常小)，权重变化太大。解决：mini batch，一批样本梯度累加到一起，更新一次。

