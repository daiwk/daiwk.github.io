---
layout: post
category: "ml"
title: "特征工程"
tags: [特征工程, sk-dist, ]
---

目录

<!-- TOC -->

- [0. 工具包](#0-工具包)
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
- [6. tf的特征工程](#6-tf的特征工程)
- [7. paddle的ctr特征demo](#7-paddle的ctr特征demo)
- [sk-dist](#sk-dist)

<!-- /TOC -->


## 0. 工具包

代码: [https://github.com/abhayspawar/featexp](https://github.com/abhayspawar/featexp)
说明: [业界 \| 如何达到Kaggle竞赛top 2%？这里有一篇特征探索经验帖](https://mp.weixin.qq.com/s/SDCon7Uy-E4NlLN8JxzkFg)


## 1. 概述

参考[https://www.cnblogs.com/jasonfreak/p/5448385.html](https://www.cnblogs.com/jasonfreak/p/5448385.html)

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

简单的方差的方法

`\[
D(x) = E(x^2)-[E(x)]^2
\]`

所以用python来写就是：

```python
N = len(nlist)
narray = numpy.array(nlist)
sum1 = narray.sum()
narray2 = narray * narray
sum2 = narray2.sum()
mean = sum1 / N
var = sum2 / N - mean ** 2
stdv = math.sqrt(var)

print mean
print var
print stdv
```

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

比如有10个特征，然后你有一个模型a用这10个特征来预测xxx

wrapper我理解就是10个特征过这个模型，随机扔掉一些，看单特征的效果（类似所谓的的特征重要度）；

embedded我理解是另外拿一个模型，目标可能不一定是你要预测的xxx，通过那个模型，来给每个特征学一个权重，然后筛选特征，剩下的特征再用a去走（例如模型是个复杂dnn，可以先用lr/gbdt，输入这些特征，去学某个label，然后学到权重用来筛选特征吧）

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

## 6. tf的特征工程

官方文档：[https://www.tensorflow.org/guide/feature_columns?hl=zh-cn](https://www.tensorflow.org/guide/feature_columns?hl=zh-cn)

参考[https://blog.csdn.net/cjopengler/article/details/78161748](https://blog.csdn.net/cjopengler/article/details/78161748)

同样地，参考[https://zhuanlan.zhihu.com/p/41663141](https://zhuanlan.zhihu.com/p/41663141)


通过调用tf.feature_column模块来创建feature columns。有两大类feature column

+ 一类是生成dense tensor的Dense Column。包括：
  + numerical_column
  + indicator_column：one-hot或者multi-hot的sparsetensor
  + embedding_column：每个bucket对应一个emb向量，如果是multi-hot可以指定combine方式，如sum，avg等
  + shared_embedding_columns：有多个特征可能需要共享相同的embeding映射空间，比如用户历史行为序列中的商品ID和候选商品ID
+ 另一类是生成sparse tensor的Categorical Column。包括：
  + categorical_column_with_identity：和bucketize类似，也是用一个单个的唯一值表示bucket。
  + categorical_column_with_vocabulary_file：通过一个文件的名字来one-hot
  + categorical_column_with_vocabulary_list：通过一个list的名字来one-hot
  + categorical_column_with_hash_bucket：通过hash的方式来得到最终的类别ID
  + crossed_column：特征先进行笛卡尔积，再hash
  + weighted_categorical_column：
+ 还有bucketized_column，可以生成Categorical Column也可以生成Dense Column：把numeric column的值按照提供的边界（boundaries)离散化为多个值

使用indicator_column能把categorical column得到的稀疏tensor转换为one-hot或者multi-hot形式的稠密tensor

demo：

```python
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder
def test_shared_embedding_column_with_hash_bucket():
    color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]],
                  'color2': [[2], [5], [-1], [0]]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    color_column2 = feature_column.categorical_column_with_hash_bucket('color2', 7, dtype=tf.int32)
    color_column_tensor2 = color_column2._get_sparse_tensors(builder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('not use input_layer' + '_' * 40)
        print(session.run([color_column_tensor.id_tensor]))
        print(session.run([color_column_tensor2.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_embed = feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
    print(type(color_column_embed))
    color_dense_tensor = feature_column.input_layer(color_data, color_column_embed)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run(color_dense_tensor))

test_shared_embedding_column_with_hash_bucket()
```

还有：

```python
def test_categorical_column_with_vocabulary_list():
    color_data = {'color': [['R', 'R'], ['G', 'R'], ['B', 'G'], ['A', 'A']]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )

    color_column_tensor = color_column._get_sparse_tensors(builder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_identy = feature_column.indicator_column(color_column)
    color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))

test_categorical_column_with_vocabulary_list()
```

对于多个取值特征进行feature crossing。。。参考[https://github.com/Lapis-Hong/wide_deep/blob/master/python/lib/dataset.py#L152](https://github.com/Lapis-Hong/wide_deep/blob/master/python/lib/dataset.py#L152)的掉渣天的代码：

```python

_CSV_COLUMNS = [
        "ad_account_id", "education", "bid"
        "show"]

_CSV_COLUMN_DEFAULTS = [
        ['-1'], ['-1'], [0.0], 
        [0.0]]

ad_product_id = tf.feature_column.categorical_column_with_hash_bucket(
      'ad_product_id', hash_bucket_size=12000)

base_columns = [
      ad_account_id, ]

crossed_columns = [
    tf.feature_column.crossed_column(
          ['ad_product_id', 'education_cross2'],
          hash_bucket_size=2000),
]

wide_columns = base_columns + crossed_columns

deep_columns = [
      bid,
]
def input_fn(data_file, num_epochs, shuffle, batch_size):
  def parse_csv(value):
    tf.logging.info('Parsing {}'.format(data_file))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    # features = {"bid": csv_decode_obj, "show": csv_decode_obj, ...}
    features["education_cross2"] = tf.string_split(columns[5:6], delimiter=":").values
    labels = features.pop('show')
    return features, labels
```

然而要batch的时候，还是会有点问题的，所以需要改成(注意！！padded_shape如果传入第一个元素的是个字典，会按key排序的，也就是"bid"会排在"age"的后面！！)：

```python
_CSV_COLUMNS = [
        "ad_account_id", "education", "age","bid",
        "show"]

_CSV_COLUMN_DEFAULTS = [
        ['-1'], ['-1'], ['-1'], [0.0], 
        [0.0]]

ad_product_id = tf.feature_column.categorical_column_with_hash_bucket(
      'ad_product_id', hash_bucket_size=12000)

base_columns = [
      ad_account_id, ]

crossed_columns = [
    tf.feature_column.crossed_column(
          ['ad_product_id', 'education'],
          hash_bucket_size=2000),
]

wide_columns = base_columns + crossed_columns

deep_columns = [
      bid,
]
def input_fn(data_file, num_epochs, shuffle, batch_size):
  def parse_csv(value):
    tf.logging.info('Parsing {}'.format(data_file))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    # features = {"bid": csv_decode_obj, "show": csv_decode_obj, ...}
    features["education"] = tf.string_split(columns[5:6], delimiter=":").values
    features["age"] = tf.string_split(columns[6:7], delimiter=":").values
    labels = features.pop('show')
    return features, labels

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  g_features = _CSV_COLUMNS[:-1]
  padded_dict = {k: [] for k in g_features}
  padded_dict["age"] = [-1] # 如果设成某个数，比如我想pad成长度5的之类的。。会出现『Attempted to pad to a smaller size than the input element.』的错。。比较蛋疼
  padded_dict["education"] = [-1]
  if mode == "pred": 
      padded_dict = (padded_dict, [])
  else:
      padded_dict = (padded_dict, [])
  dataset = dataset.padded_batch(batch_size, padded_shapes=padded_dict)#, drop_remainder=True)#.filter(lambda fea,lab: tf.equal(tf.shape(lab)[0], batch_size))
  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
##  dataset = dataset.batch(batch_size) ## 如果有变长的，就不能用这个啦，要用上面的padded_batch
```

另外，在[https://github.com/daiwk/grace_t/tree/master/python/grace_t/basic_demos](https://github.com/daiwk/grace_t/tree/master/python/grace_t/basic_demos)这里有些实践和尝试。。

## 7. paddle的ctr特征demo

参考[https://zhuanlan.zhihu.com/p/32699487](https://zhuanlan.zhihu.com/p/32699487)

参考[https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/ctr/preprocess.py](https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/ctr/preprocess.py)

```python
class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return list(map(len, self.dicts))
```

```python
class ContinuousFeatureGenerator:
    """
    Normalize the integer features to [0, 1] by min-max normalization
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > continous_clip[i]:
                            val = continous_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])
```

## sk-dist

[将sklearn训练速度提升100多倍，美国「返利网」开源sk-dist框架](https://mp.weixin.qq.com/s/mjQuJoP4VIZ2yHToX19aMA)

[https://github.com/Ibotta/sk-dist](https://github.com/Ibotta/sk-dist)
