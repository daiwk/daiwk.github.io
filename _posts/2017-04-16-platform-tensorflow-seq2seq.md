---
layout: post
category: "platform"
title: "tf的seq2seq"
tags: [seq2seq, tensorflow]
---

使用tf，但独立于tensorflow/models之外的一个seq2seq的框架：

主页：[https://google.github.io/seq2seq/](https://google.github.io/seq2seq/)

## 1. 简介

对应的论文：
[Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf)

## 2. 安装

首先安装python-tk（python的gui）

```
apt-get install python-tk
```

然后设置matplotlib的[backend](http://matplotlib.org/faq/usage_faq.html#what-is-a-backend)

```
echo "backend : Agg" >> $HOME/.config/matplotlib/matplotlibrc
```

然后安装seq2seq

```
cd seq2seq
pip install -e .
```

测试一下

```
python -m unittest seq2seq.test.pipeline_test
```

## 3. 基本概念



## 4. 示例：nmt

经典：
+ [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
+ [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial (Neubig et al.)](https://arxiv.org/pdf/1703.01619.pdf)
+ [Tensorflow Sequence-To-Sequence Tutorial](https://www.tensorflow.org/tutorials/seq2seq)

### 4.1 数据格式

正常的分词是空格切分或者常用的分词工具（如Moses的[tokenizer.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl)、框架有[spacy](https://spacy.io/docs/usage/processing-text)/[nltk](http://www.nltk.org/api/nltk.tokenize.html)/[stanford的分词](https://nlp.stanford.edu/software/tokenizer.shtml)）。

但用于nmt时存在如下问题：
+ nmt输出的是在词上的概率分布，所以如果可能的词很多，那么会非常慢。如果vocabulary中有拼错的词和派生词，那vocabulary就可能趋向无穷大……而我们实际上可能会**人工限制vocabulary size在10,000-100,000的范围内。**
+ loved和loving来自同一词根，但却被当成完全不同的两个词来看待。

对于这种**open vocabulary**的问题一，一种解决方法就是从给定的文本中学习**subword units**。例如，loved可以被分成lov和ed，loving可以被分成lov和ing。这样，一方面可以产生新的词（unknown words），另一方面，可以缩减vocabulary size。本示例用到的就是**[Byte Pair Encoding (BPE)](https://arxiv.org/pdf/1508.07909.pdf)**

用法：

```
# Clone from Github
git clone https://github.com/rsennrich/subword-nmt
cd subword-nmt

# Learn a vocabulary using 10,000 merge operations
./learn_bpe.py -s 10000 < train.tok > codes.bpe

# Apply the vocabulary to the training file
./apply_bpe.py -c codes.bpe < train.tok > train.tok.bpe
```

结果如下，会用对不常见的词（如例子中的Nikitin）使用```@@```进行切分。

```
Madam President , I should like to draw your attention to a case in which this Parliament has consistently shown an interest. It is the case of Alexander Ni@@ ki@@ tin .
```

### 4.2 数据下载

### 4.3 小数据集：generate toy data

### 4.4 定义模型

### 4.5 训练

### 4.6 预测

### 4.7 使用beamsearch进行decode

### 4.8 基于checkpoint进行评估

### 4.9 计算BLEU