---
layout: post
category: "nlp"
title: "分词工具"
tags: [分词, 切词]
---

目录

<!-- TOC -->


<!-- /TOC -->

参考[北大开源全新中文分词工具包：准确率远超THULAC、结巴分词](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650755353&idx=3&sn=12aacb01478eb362584581383117200f&chksm=871a9567b06d1c71dba040d3614dbae2b07ad693865c18556240d7bf958a3e026a298cf26853&mpshare=1&scene=1&srcid=0113bqxHuiWfIiVCVso4KrY1&pass_ticket=GbqnkzYDgSDQxJoviNYzckA8ZJ6bULsWpoyug4CHgCsT0B80C5nEC38bRj4CywCT#rd)

github：[https://github.com/lancopku/PKUSeg-python](https://github.com/lancopku/PKUSeg-python)

特点：

+ **高分词准确率**：相比于其他的分词工具包，该工具包在不同领域的数据上都大幅提高了分词的准确度。根据北大研究组的测试结果，pkuseg 分别在示例数据集（MSRA 和 CTB8）上降低了 79.33% 和 63.67% 的分词错误率。
+ **多领域分词**：研究组训练了多种不同领域的分词模型。根据待分词的领域特点，用户可以自由地选择不同的模型。
+ **支持用户自训练模型**：支持用户使用全新的标注数据进行训练。

