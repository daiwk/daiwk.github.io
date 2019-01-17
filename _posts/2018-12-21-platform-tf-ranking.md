---
layout: post
category: "platform"
title: "tf-ranking"
tags: [tf-ranking, ]
---

目录

<!-- TOC -->

- [安装](#安装)

<!-- /TOC -->

参考[TF-Ranking：为 Learning-to-Rank 打造的可扩展 TensorFlow 库](https://mp.weixin.qq.com/s?__biz=MzU1OTMyNDcxMQ==&mid=2247485361&idx=1&sn=fe8dad5e5dfe8baabe4d60af68b32415&chksm=fc184cf9cb6fc5ef562e20ade54be893cf5f8ffbe89c07ed9e5d6b25cac9bb17fa7a73b52865&mpshare=1&scene=1&srcid=0118obA2nmTOwM9mkfe8HZmY&pass_ticket=yoIK672aXk4WPiJRK3zkCxK5C5wwnua1%2B%2F115s%2FKJyXjdHQlvctIkGZpDsP%2FPVPZ#rd)

[https://github.com/tensorflow/ranking](https://github.com/tensorflow/ranking)

[https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/examples](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/examples)

[TF-Ranking: Scalable TensorFlow Library for Learning-to-Rank](https://arxiv.org/abs/1812.00073)

## 安装

```shell
bazel build //tensorflow_ranking/tools/pip_package:build_pip_package
bazel-bin/tensorflow_ranking/tools/pip_package/build_pip_package /tmp/ranking_pip
```

这样就产生了一个whl文件（要求tf1.12.0+）

TF-Ranking 能通过嵌入和扩展至数百亿个训练实例来处理稀疏特征（如原始文本）

TF-Ranking 支持许多常用的排名指标，包括平均倒序排名 (MRR) 和标准化折扣累积收益 (NDCG)。

TF-Ranking 支持一种新颖的评分机制，可以对多个项目（例如网页）执行联合评分，这是对传统评分范例（对单个项目进行独立评分）的扩展。关于多项目评分，我们经常面临的一个挑战是，难以判断需将哪些项目进行分组并在子组中进行评分。然后，累计每个项目的评分并用于排序。为了让用户免于接触这些复杂原理，TF-Ranking 提供了一个**List-In-List-Out (LILO) API**，将所有逻辑封装于导出的 TensorFlow 模型内。