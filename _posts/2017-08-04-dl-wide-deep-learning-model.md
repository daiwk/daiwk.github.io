---
layout: post
category: "dl"
title: "Wide & Deep Learning Model"
tags: [Wide & Deep Learning Model, ]
---


## 传统方法：

### LR+GBDT

[CTR预估中GBDT与LR融合方案](http://www.cbdio.com/BigData/2015-08/27/content_3750170.htm)

LR映射后的函数值就是CTR的预估值。这种线性模型很容易并行化，处理上亿条训练样本不是问题，但线性模型学习能力有限，需要大量特征工程预先分析出有效的特征、特征组合，从而去间接增强LR 的非线性学习能力。

LR模型中的特征组合很关键，但又无法直接通过特征笛卡尔积 解决，只能依靠人工经验，耗时耗力同时并不一定会带来效果提升。

GBDT又叫MART（Multiple Additive Regression Tree)，GBDT中的树是**回归树**（不是分类树），GBDT用来做回归预测，调整后也可以用于分类。每次迭代都在**减少残差的梯度方向新建立一颗决策树**，迭代多少次就会生成多少颗决策树。GBDT的思想使其具有天然优势，可以发现多种有区分性的特征以及特征组合，决策树的路径可以直接作为LR输入特征使用，省去了人工寻找特征、特征组合的步骤。这种通过GBDT生成LR特征的方式（GBDT+LR），业界已有实践（Facebook，Kaggle-2014），且效果不错。

回归树总体流程类似于分类树，区别在于，回归树的每一个节点都会得一个预测值，以年龄为例，该预测值等于属于这个节点的所有人年龄的平均值。分枝时穷举每一个feature的每个阈值找最好的分割点，但衡量最好的标准不再是最大熵，而是最小化平方误差。也就是被预测出错的人数越多，错的越离谱，平方误差就越大，通过最小化平方误差能够找到最可靠的分枝依据。分枝直到每个叶子节点上人的年龄都唯一或者达到预设的终止条件(如叶子个数上限)，若最终叶子节点上人的年龄不唯一，则以该节点上所有人的平均年龄做为该叶子节点的预测年龄。

参考[http://blog.csdn.net/puqutogether/article/details/44593647](http://blog.csdn.net/puqutogether/article/details/44593647)

<html>
<br/>

<img src='../assets/regression-tree.png' style='max-height: 350px'/>
<br/>

</html>




[Wide & deep learning for recommender systems](https://arxiv.org/pdf/1606.07792.pdf)

用于ctr预估[https://github.com/PaddlePaddle/models/tree/develop/ctr](https://github.com/PaddlePaddle/models/tree/develop/ctr)
