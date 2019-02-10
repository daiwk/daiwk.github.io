---
layout: post
category: "nlp"
title: "tagspace"
tags: [tagspace, ]
---

目录

<!-- TOC -->

- [概述](#概述)

<!-- /TOC -->


## 概述

[#TagSpace: Semantic Embeddings from Hashtags](https://research.fb.com/wp-content/uploads/2014/09/tagspace-semantic-embeddings-from-hashtags.pdf)

[https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags/](https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags/)

<html>
<br/>

<img src='../assets/tagspaces.png' style='max-height: 350px'/>
<br/>

</html>



假设一句话有l个词，总词典大小是N，那么大的emb矩阵就是Nxd，而对于这句话来讲，就是一个lxd的矩阵，然后用same padding，并用1D卷积，详细地说，就是先padding成一个(l+K-1)xd的矩阵，然后有H个Kxd的卷积核，这样得到的结果就是一个(l+K-1-K+1)x(d-d+1)xH=lx1xH=lxH的矩阵了，之所以叫same padding，就是因为一开始的第一维是l，最终的第一维仍是l。

对于maxpooling来说，输入是lxH，按照论文和图中的，应该是l个数取max，得到1xH，但下面的tf的实现，好像不是这样的呢。。。paddle的实现用的是[https://daiwk.github.io/posts/nlp-nmt.html#8-%E5%85%B6%E4%BB%96](https://daiwk.github.io/posts/nlp-nmt.html#8-%E5%85%B6%E4%BB%96)这个的sequence_conv_pool函数，实现参考

参考tf代码[https://github.com/flrngel/TagSpace-tensorflow/blob/master/model.py](https://github.com/flrngel/TagSpace-tensorflow/blob/master/model.py)

```python
      doc_embed = tflearn.embedding(doc, input_dim=N, output_dim=d)
      self.lt_embed = lt_embed = tf.Variable(tf.random_normal([tN, d], stddev=0.1)) # 在卷积这步这个变量没啥用

      net = tflearn.conv_1d(doc_embed, H, K, activation='tanh')# conv_1d默认是same padding，卷积核是Kxd，有H个卷积核
      net = tflearn.max_pool_1d(net, K) # max_pool_1d默认也是same pooling
      net = tflearn.tanh(net)
      self.logit = logit = tflearn.fully_connected(net, d, activation=None)
```

参考paddle代码：[https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleRec/tagspace/net.py](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleRec/tagspace/net.py)

```python
    text_emb = nn.embedding(
            input=text, size=[vocab_text_size, emb_dim], param_attr="text_emb")
    pos_tag_emb = nn.embedding(
            input=pos_tag, size=[vocab_tag_size, emb_dim], param_attr="tag_emb")
    neg_tag_emb = nn.embedding(
            input=neg_tag, size=[vocab_tag_size, emb_dim], param_attr="tag_emb")

    conv_1d = fluid.nets.sequence_conv_pool(
            input=text_emb,
            num_filters=hid_dim,
            filter_size=win_size,
            act="tanh",
            pool_type="max",
            param_attr="cnn")
    text_hid = fluid.layers.fc(input=conv_1d, size=emb_dim, param_attr="text_hid")
    cos_pos = nn.cos_sim(pos_tag_emb, text_hid)
```
