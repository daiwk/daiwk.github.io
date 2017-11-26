---
layout: post
category: "ml"
title: "特征工程"
tags: [特征工程, ]
---

目录

<!-- TOC -->

- [1. 概述](#1-概述)
    - [1.1 数据预处理(单一特征)](#11-数据预处理单一特征)
    - [1.2 特征选择(多特征)](#12-特征选择多特征)
    - [1.3 降维(多特征)](#13-降维多特征)
- [2. 数据预处理](#2-数据预处理)
    - [2.1 特征离散化的好处](#21-特征离散化的好处)
    - [2.2 特征常见统计指标](#22-特征常见统计指标)
- [3. 特征选择](#3-特征选择)
    - [3.1 特征间的统计指标](#31-特征间的统计指标)
    - [3.2 特征选择的形式](#32-特征选择的形式)
- [4. 降维](#4-降维)
- [5. sklearn小技巧](#5-sklearn小技巧)

<!-- /TOC -->

参考[https://www.cnblogs.com/jasonfreak/p/5448385.html](https://www.cnblogs.com/jasonfreak/p/5448385.html)

## 1. 概述

本质是一项工程活动，目的是最大限度地从原始数据中提取特征以供算法和模型使用。

<html>
<br/>

<img src='../assets/feature_engineering.jpg' style='max-height: 450px;max-width:500px'/>
<br/>

</html>

### 1.1 数据预处理(单一特征)

<html>
<center>
<table border="2" cellspacing="0" cellpadding="6" rules="all" frame="border">

<thead>
<tr>
<th scope="col" class="left">库</th>
<th scope="col" class="left">所属方法</th>
<th scope="col" class="left">说明</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left">preprocessing.StandardScaler</td>
<td class="left">无量纲(dimension)化</td>
<td class="left">标准化，基于特征矩阵的列，将特征值转换至服从标准正态分布</td>
</tr>

<tr>
<td class="left">preprocessing.MinMaxScaler</td>
<td class="left">无量纲(dimension)化</td>
<td class="left">区间缩放，基于最大最小值，将特征值转换到[0, 1]区间上</td>
</tr>

<tr>
<td class="left">preprocessing.Normalizer</td>
<td class="left">归一化</td>
<td class="left">基于特征矩阵的行，将样本向量转换为『单位向量』</td>
</tr>

<tr>
<td class="left">preprocessing.Binarizer</td>
<td class="left">二值化</td>
<td class="left">基于给定阈值，将定量特征按阈值划分</td>
</tr>

<tr>
<td class="left">preprocessing.OneHotEncoder</td>
<td class="left">one-hot编码</td>
<td class="left">将定性数据转换为定量数据 </td>
</tr>

<tr>
<td class="left">preprocessing.Imputer</td>
<td class="left">缺失值计算</td>
<td class="left">计算缺失值，填充为均值等</td>
</tr>

<tr>
<td class="left">preprocessing.PolynomialFeatures</td>
<td class="left">多项式数据转换</td>
<td class="left">多项式数据转换</td>
</tr>

<tr>
<td class="left">preprocessing.FunctionTransformer</td>
<td class="left">自定义单元数据转换</td>
<td class="left">使用单变元的函数来转换数据</td>
</tr>

</tbody>
</table></center>
</html>

### 1.2 特征选择(多特征)

<html>
<center>
<table border="2" cellspacing="0" cellpadding="6" rules="all" frame="border">

<thead>
<tr>
<th scope="col" class="left">库</th>
<th scope="col" class="left">所属方法</th>
<th scope="col" class="left">说明</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left">feature_selection.VarianceThreshold</td>
<td class="left">Filter</td>
<td class="left">方差选择法</td>
</tr>

<tr>
<td class="left">feature_selection.SelectKBest</td>
<td class="left">Filter</td>
<td class="left">常用函数
<br/>
<li>pearson相关系数（scipy.stats.pearsonr）</li>
<li>卡方检验(sklearn.feature_selection.chi2)</li>
<li>互信息(minepy.MINE)</li>
</td>
</tr>

<tr>
<td class="left">feature_selection.RFE</td>
<td class="left">Wrapper</td>
<td class="left">特征递归消除法（递归地训练基模型[例如，sklearn.linear_model.LogisticRegression]，将权值系数较小的特征从特征集合中消除）</td>
</tr>

<tr>
<td class="left">feature_selection.SelectFromModel</td>
<td class="left">Embeded</td>
<td class="left">训练基模型，选择权值系数较高的特征
<li>基于惩罚项的特征选择(结合sklearn.linear_model.LogisticRegression)</li>
<li>基于树模型的特征选择(结合sklearn.ensemble.GradientBoostingClassifier)</li>
</td>
</tr>
</tbody>
</table></center>
</html>

### 1.3 降维(多特征)

<html>
<center>
<table border="2" cellspacing="0" cellpadding="6" rules="all" frame="border">

<thead>
<tr>
<th scope="col" class="left">库</th>
<th scope="col" class="left">所属方法</th>
<th scope="col" class="left">说明</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left">decomposition.PCA</td>
<td class="left">  PCA</td>
<td class="left">主成分分析，为了让映射后的样本有最大的发散性，是一种无监督的降维方法。</td>
</tr>

<tr>
<td class="left">lda.LDA</td>
<td class="left">  LDA</td>
<td class="left">线性判别分析法，为了让映射后的样本有最好的分类性能，是一种有监督的降维方法。</td>
</tr>

</tbody>
</table></center>
</html>

## 2. 数据预处理

### 2.1 特征离散化的好处

+ 稀疏向量内积乘法运算速度快，计算结果方便存储，易扩展
+ 离散化后的特征对异常数据有很强的鲁棒性
+ 单变量离散化为n个后，每个特征有独立的权重，相当于多个特征，引入非线性
+ 离散化后再进行特征组合，如1->m，1->n，那组合就有m*n
+ 离散化后，模型会更稳定。和第二点类似。

### 2.2 特征常见统计指标

+ 数据量大小，最大最小值，distinct(特征下不同值的个数)
+ 平均值、中位数、众数
+ 方差、变异系数(Coefficient of Variation，标准差/平均值，变异系数是一个无量纲量，因此在比较两组量纲不同或均值不同的数据时，应该用变异系数而不是标准差来作为比较的参考)
+ 熵(不确定性的度量，不确定性越强、越混乱，熵越大)、偏度(skewness，所有取值分布的对称性)、峰度(Kurtosis，所有取值分布形态陡缓程度)

## 3. 特征选择

两个考虑点：

+ 特征是否发散：如果不发散，即方差接近0，说明样本在这个特征上基本没差异，这个特征对于样本的区分没什么作用==>特征间彼此相关性弱的是好的特征
+ 特征与目标的相关性：除方差法，其他方法均从相关性角度考虑。==>与目标相关性强的特征是好的特征


### 3.1 特征间的统计指标

+ 连续x连续：pearson相关系数、spearman相关系数
+ 分类x分类：卡方独立性检验(`\(X^2\)`越小，说明变量越独立，越大越相关)
+ 连续x分类：
    + 针对两组特征：t检验(检测数据的准确度，系统误差，类似bias，和真实值的差距，z检验也类似，但z检验需要知道总体方差，不容易知道，所以常用t检验)、F检验(检测数据的精密度，偶然误差，类似variance，方差)
    + 针对多于两组特征：ANOVA
+ 最大信息系数（Mutual information and maximal information coefficient，MIC）：缺点：值域[0,1]，当零假设不成立时，MIC的统计会受影响。
+ 信息增益(待分类的集合的熵和选定某个特征的条件熵之差，参考[https://www.cnblogs.com/fantasy01/p/4581803.html](https://www.cnblogs.com/fantasy01/p/4581803.html))
+ 多特征距离的计算（欧式距离、标准化欧式距离、余弦、jaccard距离、KL散度）

### 3.2 特征选择的形式

根据特征选择的形式，可以分为以下三种方法：
+ Filter: 过滤法，按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征
+ Wrapper：包装法，根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。中间的搜索方法有完全搜索、启发式搜索、随机搜索
+ Embedded：嵌入法，先使用某些机器学习算法或者模型进行训练 ，得到和个特征的权值系数，根据系数从大到小特征特征。
    + 正则化：L1：不稳定，受噪声影响大，因为表达能力差的特征系数为0；L2：更稳定，因为表达能力强的特征的系数非零
    + 树模型



## 4. 降维

pca.explained_variaance_ratio_：percentage of variance explained by each of the selected components. 


## 5. sklearn小技巧

参考 [http://www.cnblogs.com/jasonfreak/p/5448462.html](http://www.cnblogs.com/jasonfreak/p/5448462.html)

<html>
<center>
<table border="2" cellspacing="0" cellpadding="6" rules="all" frame="border">

<thead>
<tr>
<th scope="col" class="left">包</th>
<th scope="col" class="left">类或方法</th>
<th scope="col" class="left">说明</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left">sklearn.pipeline</td>
<td class="left">Pipeline</td>
<td class="left">流水线处理</td>
</tr>

<tr>
<td class="left">sklearn.pipeline</td>
<td class="left">FeatureUnion</td>
<td class="left">并行处理</td>
</tr>

<tr>
<td class="left">sklearn.grid_search</td>
<td class="left">GridSearchCV</td>
<td class="left">网格搜索调参</td>
</tr>


</tbody>
</table></center>
</html>
