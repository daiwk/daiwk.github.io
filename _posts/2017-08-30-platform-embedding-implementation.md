---
layout: post
category: "platform"
title: "各框架的embedding实现"
tags: [embedding,]
---

## keras

[https://keras.io/layers/embeddings/](https://keras.io/layers/embeddings/) keras的embedding是参考[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf)这个实现的，类似对输入数据进行dropout。

## paddle

[https://github.com/PaddlePaddle/Paddle/issues/2910](https://github.com/PaddlePaddle/Paddle/issues/2910)


## 应用

[DNN在搜索场景中的应用](http://www.cnblogs.com/hujiapeng/p/6236857.html)

wide model

  a. id feature: item_id, seller_id，学习已经出现过的商品，卖家在训练数据上的表现。
  b. id cross feature: user_id x item_id , user_id x seller_id

连续值统计特征是非常有用的特征，Google的模型是把embedding向量和统计特征放到同一个DNN网络中学习，但实验发现这样会削弱统计特征的作用。我们为统计特征专门又组建了一个包含2个隐层的网路，并且为了增强非线性效果，激活函数从RELU改为TanH/Sigmiod。

 

deep model

  a. 首先需要把离散特征（item_id，item_tag, user_id，user_tag，query_tag）embeding成连续特征。
  b. 将embedding后的向量作为DNN的输入。考虑到最终线上预测性能的问题，目前我们的DNN网络还比较简单，只有1到2个隐层。

整体模型使用三层全连接层用于sparse+dense特征表征学习，再用两层全连接层用于点击/购买与否分类的统一深度学习模型解决方案：
  第一层为编码层，包含商品编码，店家编码，类目编码，品牌编码，搜索词编码和用户编码。

在普适的CTR场景中，用户、商品、查询等若干个域的特征维度合计高达几十亿，假设在输入层后直接连接100个输出神经元的全连接层，那么这个模型的参数规模将达到千亿规模。直接接入全连接层将导致以下几个问题：1. 各个域都存在冷门的特征，这些冷门的特征将会被热门的特征淹没，基本不起作用，跟全连接层的连接边权值会趋向于0，冷门的商品只会更冷门。2. 模型的大小将会非常庞大，超过百G，在训练以及预测中都会出现很多工程上的问题。为了解决上述两个问题，本文引入了紫色编码层，具体分为以下两种编码方式：1. 随机编码 2. 挂靠编码，下面将对以上两种编码方式进行详细的描述。
