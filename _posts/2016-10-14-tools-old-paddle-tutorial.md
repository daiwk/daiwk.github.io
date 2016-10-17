---
layout: post
category: "tools"
title: "老Paddle使用整理"
tags: [老paddle,]
---

# 老Paddle使用整理

以**bi-lstm+crf**进行品牌词识别为例，对老paddle的使用进行总结。

## local版本

### 预处理

#### 使用proto sequence格式
假定一条样本为中文句子**“百度 公司 创立于 2000年”**，样本的类别为“0”。假定“百度”在词表中的id为23，“公司”为35，“创立于”为10，“2000年”为87，词表大小为23984，共有3个类别。
首先我们将其转换成文本格式（label; slot_id word_id1 word_id2 word_id3 word_id4……）【注意，slotid没有实际意义，只是一个编号，表示有多少维特征，从0开始递增】:

>0;0 23 35 10 87;

而这里我们要需要进行序列标注，所以在wordid这一维特征之外，还要用到别的两个特征，总共如下：

- wordid：假设有
- 词性：pos，假设有110个类别
- 占位符：全部写0

比如，一个单词有8个汉字，那么，我们转化为：
>0;0 1383 2523 4396 1253 3967 4333 490 613;1 48 94 94 32 86 17 70 25;2 0 0 0 0 0 0 0 0;

接下来，使用txt2proto工具进行转换：
>cat INPUT_FILE | txt2proto OUTPUT_FILE "1" "23984 3"


#### 使用pyDataProvider格式


## 集群版本