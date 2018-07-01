---
layout: post
category: "dl"
title: "youtube视频推荐系统"
tags: [youtube视频推荐系统, ]
---

目录


<!-- TOC -->

- [候选生成网络（Candidate Generation Network）](#%E5%80%99%E9%80%89%E7%94%9F%E6%88%90%E7%BD%91%E7%BB%9C%EF%BC%88candidate-generation-network%EF%BC%89)
    - [4.2 Modeling Expected Watch Time](#42-modeling-expected-watch-time)
- [代码实现](#%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0)

<!-- /TOC -->

参考[http://www.sohu.com/a/155797861_465975](http://www.sohu.com/a/155797861_465975)

参考[https://zhuanlan.zhihu.com/p/25343518](https://zhuanlan.zhihu.com/p/25343518)

[Deep neural networks for youtube recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)

YouTube是世界上最大的视频上传、分享和发现网站，YouTube推荐系统为超过10亿用户从不断增长的视频库中推荐个性化的内容。整个系统由两个神经网络组成：**候选生成网络**和**排序网络**。候选生成网络从**百万量级**的视频库中生成**上百个**候选，排序网络对候选进行打分排序，输出**排名最高的数十个结果**。

## 候选生成网络（Candidate Generation Network）

候选生成网络将推荐问题建模为一个**类别数极大的多分类问题**：对于一个Youtube用户，使用其观看历史（视频ID）、搜索词记录（search tokens）、人口学信息（如地理位置、用户登录设备）、二值特征（如性别，是否登录）和连续特征（如用户年龄）等，对视频库中所有视频进行多分类，得到每一类别的分类结果（即每一个视频的推荐概率），最终输出概率较高的几百个视频。===>即，【使用**用户特征**，对所有视频进行分类，得到**和这个用户最相关的几百个候选结果。**】

### 4.2 Modeling Expected Watch Time

训练用的是logistic regression加上cross-entropy，

假设第i个正样本的播放时长是`\(T_i\)`，使用weighted logistic regression，将正样本的权重设为播放时长，而负样本的权重设为1，这样，假设总共有N个样本，有k个被点击了，就相当于有了`\(\sum T_i\)`个正样本，N-k个负样本。所以odds（注：一个事件的几率odds指该事件发生与不发生的概率比值）就是正样本数/负样本数=`\(\frac{\sum T_i}{N-k}\)`。

而实际中，点击率P很低，也就是k很小，而播放时长的期望是`\(E(T)=\frac{\sum T_i}{N}\)`，所以`\(E(T)\)`约等于`\(E(T)(1+P)\)`，约等于odds，即`\(\frac{\sum T_i}{N-k}\)`

最后在inference的serving中，直接使用`\(e^{Wx+b}\)`来产出odds，从而近似expected watch time。


<html>
<br/>

<img src='../assets/youtube-dnn-recsys-architecture.png' style='max-height: 350px;max-width:500px'/>
<br/>

</html>

## 代码实现

[https://github.com/ogerhsou/Youtube-Recommendation-Tensorflow/blob/master/youtube_recommendation.py](https://github.com/ogerhsou/Youtube-Recommendation-Tensorflow/blob/master/youtube_recommendation.py)


关于数据集：

[https://github.com/ogerhsou/Youtube-Recommendation-Tensorflow/commit/e92bac1b8b5deb0e93e996b490561baaea60bae8](https://github.com/ogerhsou/Youtube-Recommendation-Tensorflow/commit/e92bac1b8b5deb0e93e996b490561baaea60bae8)

使用的是[https://github.com/facebookresearch/fastText/blob/master/classification-example.sh](https://github.com/facebookresearch/fastText/blob/master/classification-example.sh)

数据格式：

```shell
cut -d' ' -f 1 ./data/dbpedia.train | sort | uniq -c
40000 __label__1
40000 __label__10
40000 __label__11
40000 __label__12
40000 __label__13
40000 __label__14
40000 __label__2
40000 __label__3
40000 __label__4
40000 __label__5
40000 __label__6
40000 __label__7
40000 __label__8
40000 __label__9
```

在init_data函数中，给每个__label__xx编了个号，如：

```shell
__label__6 0
__label__12 1
__label__14 2
__label__7 3
__label__2 4
__label__5 5
__label__10 6
__label__13 7
__label__3 8
__label__1 9
__label__8 10
__label__11 11
__label__9 12
__label__4 13
```

然后read_data的时候，y就用这个编号来表示（假装是时长）：

```python
y.append(label_dict[line[0]])
```

而使用的是nce_loss(参考[https://daiwk.github.io/posts/knowledge-tf-usage.html#tfnnnceloss](https://daiwk.github.io/posts/knowledge-tf-usage.html#tfnnnceloss))：

```python
ce_weights = tf.Variable(
    tf.truncated_normal([n_classes, n_hidden_1],
                        stddev=1.0 / math.sqrt(n_hidden_1)))
nce_biases = tf.Variable(tf.zeros([n_classes]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=y_batch,
                     inputs=pred,
                     num_sampled=10,
                     num_classes=n_classes))

cost = tf.reduce_sum(loss) / batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
out_layer = tf.matmul(pred, tf.transpose(nce_weights)) + nce_biases
```

模型预测的是时长(等有时间了再细看呢！！)：

```python
    # Test model
    correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.reshape(y_batch, [batch_size]))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```