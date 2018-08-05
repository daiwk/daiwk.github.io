---
layout: post
category: "nlp"
title: "文本摘要"
tags: [文本摘要, text summary]
---

目录

<!-- TOC -->

- [抽取式](#抽取式)
    - [基于排序的方法](#基于排序的方法)
        - [基于启发式规则](#基于启发式规则)
        - [基于图排序的方法](#基于图排序的方法)
            - [基于PageRank算法或相似算法](#基于pagerank算法或相似算法)
            - [textrank算法的基本思想](#textrank算法的基本思想)
    - [基于有监督学习的方法](#基于有监督学习的方法)
    - [基于神经网络的方法](#基于神经网络的方法)
    - [基于整数线性规划（ILP）的方法](#基于整数线性规划ilp的方法)
    - [基于次模函数的方法](#基于次模函数的方法)
    - [摘要句子排序](#摘要句子排序)
- [生成式摘要](#生成式摘要)
    - [语料](#语料)
    - [1. Complex Attention Model](#1-complex-attention-model)
    - [2. Simple Attention Model](#2-simple-attention-model)
    - [3. Attention-Based Summarization(ABS)](#3-attention-based-summarizationabs)
    - [4. ABS+](#4-abs)
    - [5. Recurrent Attentive Summarizer(RAS)](#5-recurrent-attentive-summarizerras)
    - [6. big-words-lvt2k-1sent模型](#6-big-words-lvt2k-1sent模型)
    - [7. words-lvt2k-2sent-hieratt模型](#7-words-lvt2k-2sent-hieratt模型)
    - [8. feats-lvt2k-2sent-ptr模型](#8-feats-lvt2k-2sent-ptr模型)
    - [9. COPYNET](#9-copynet)
    - [10. MRT+NHG](#10-mrtnhg)
    - [效果](#效果)

<!-- /TOC -->


参考抽取式摘要1、2：

[https://blog.csdn.net/qq_32458499/article/details/78659372](https://blog.csdn.net/qq_32458499/article/details/78659372)

和

[https://blog.csdn.net/qq_32458499/article/details/78664199](https://blog.csdn.net/qq_32458499/article/details/78664199)

## 抽取式

从文档中抽取已有句子形成摘要。实现简单，能保证句子的可读性

可看作一个组合优化问题，可与语句压缩一并进行（可看作混合式方法）

### 基于排序的方法

#### 基于启发式规则

例如centroid-based method，考虑了句子包含词语权重、句子位置、句子与首句相似度等几个因素，加权作为句子的打分，然后排序。

#### 基于图排序的方法

##### 基于PageRank算法或相似算法 

步骤： 
1、构建图G=(V,E),句子作为顶点，句子之间有关系则构建边 
2、应用PageRank算法或相似算法获得每个顶点的权重 
3、基于句子权重选择句子形成摘要

如果网页T存在一个指向网页A的连接，则表明T的所有者认为A比较重要，从而把T的一部分重要性得分赋予A。这个重要性得分值为： 
PR（T）/L(T) 
其中PR（T）为T的PageRank值，L(T)为T的出链数 
则A的PageRank值为一系列类似于T的页面重要性得分值的累加。 
　 即一个页面的得票数由所有链向它的页面的重要性来决定，到一个页面的超链接相当于对该页投一票。一个页面的PageRank是由所有链向它的页面（链入页面）的重要性经过递归算法得到的。一个有较多链入的页面会有较高的等级，相反如果一个页面没有任何链入页面，那么它没有等级。

##### textrank算法的基本思想

d是阻尼系数，In(Vi)表示指向网页i的链接的网页集合，Out(Vj)指网页j中的链接存在的链接指向的网页的集合。

`\[
WS(V_i)=(1-d)+d*\sum _{V_j\in In(V_i)}\frac{w_{ji}}{\sum _{v_k \in Out(V_j)}w_{jk}}WS(V_j)
\]`

等式左边表示一个句子的权重（WS是weight_sum的缩写），右侧的求和表示每个相邻句子对本句子的贡献程度。与提取关键字的时候不同，一般认为全部句子都是相邻的，不再提取窗口。求和的分母wji表示两个句子的相似程度，分母又是一个weight_sum，而WS(Vj)代表上次迭代j的权重。整个公式是一个迭代的过程。

基于TextRank的自动文摘属于自动摘录，通过选取文本中重要度较高的句子形成文摘，其主要步骤如下： 
　　（1）预处理：将输入的文本或文本集的内容分割成句子这里写图片描述，构建图G =（V,E），其中V为句子集，对句子进行分词、去除停止词，得这里写图片描述，其中这里写图片描述是保留后的候选关键词。 
　　（2）句子相似度计算：构建图G中的边集E，基于句子间的内容覆盖率，给定两个句子si,sj，采用如下公式进行计算： 

`\[
sim(S_i,S_j)=\frac{|\{w_k|w_k \in S_i \& w_k \in S_j\}|}{log(|S_i|)+log(|S_j|)}
\]`

若两个句子之间的相似度大于给定的阈值，就认为这两个句子语义相关并将它们连接起来，即边的权值这里写图片描述 
　　（3）句子权重计算：根据公式，迭代传播权重计算各句子的得分； 
　　（4）抽取文摘句：将（3）得到的句子得分进行倒序排序，抽取重要度最高的T个句子作为候选文摘句。 
　　（5）形成文摘：根据字数或句子数要求，从候选文摘句中抽取句子组成文摘。

### 基于有监督学习的方法

+ 句子分类 
二类分类：句子是否隶属于摘要 
SVM（支持向量机）

+ 序列标注 
为每个句子打上标签 
可考虑相邻句子之间的关系 
HMM（隐马尔科夫模型），CRF（条件随机场）

+ 句子回归 
为每个句子预测一个反映重要性的分数 
SVR（支持向量回归）

### 基于神经网络的方法

+ 基于编码器-解码器框架进行单文档摘要 

编码器：先对句子编码（利用CNN），再对文档编码（利用RNN） 
解码器：输出一个0/1序列，进行句子抽取（序列标注）

+ 摘要冗余去除

去除（多文档）摘要中的冗余信息 
选择与摘要中已有句子冗余度小的句子

文本蕴涵识别技术很适合此目的，但是由于自身性能太差，无法真正应用。一般基于文本相似度来进行判断。

### 基于整数线性规划（ILP）的方法

将摘要看做一个带约束的优化问题 
基于ILP进行求解，可采用现成的ILP求解工具 
比如IBM CPLEX Optimizer 
同时进行句子抽取与冗余去除

### 基于次模函数的方法

将摘要看做一个预算约束下的次模函数最大化问题 
设计次模函数，然后利用贪心算法进行内容选取

submodular次模函数
实际上就对“边际效用递减”这个说法的形式化。就是对于一个集合函数，若，那么在S中增加一个元素所增加的收益要小于等于在S的子集中增加一个元素所增加的收益。形式化表述就是：对于函数f而言，若且，则通俗的说就是你把所有商品看成一个集合，随着你所拥有的物品数量的增加，那么你获得剩下物品所得到的满足程度越来越小。

举例说明： 
A是B的子集，则对于函数f()，如果：f(A+e)-f(A)>=f(B+e)-f(B)成立，则说f()函数是子模的。增益递减。 
例子如下： 
u={1,2,3,4,5,6,7,8} A={1,2,3} B={1,2,3,5,6} 
f(A)=|A| 集合A的个数 
所以：f(A+e)-f(A)>=f(B+e)-f(B)，例如e={3,4,5}

### 摘要句子排序

+ 句子顺序直接影响摘要可读性 

单文档摘要中句子顺序容易确定，依据句子在原文档中的顺序即可 
多文档摘要中句子顺序较难确定

+ 来自不同文档中的句子如何确定先后排序？ 

可综合考虑句子所在上下文信息进行排序。 
先确定任何两句之间的先后顺序 机器学习、深度学习 
再确定多个句子之间的整体顺序 贪心搜索


## 生成式摘要

经典。。[教机器学习摘要](https://zhuanlan.zhihu.com/p/21426100)

文本摘要问题按照文档数量可以分为单文档摘要和多文档摘要问题，按照实现方式可以分为提取式（extractive）和摘要式（abstractive）。摘要问题的特点是输出的文本要比输入的文本少很多很多，但却蕴藏着非常多的有效信息在内。有一点点感觉像是主成分分析（PCA），作用也与推荐系统有一点像，都是为了解决信息过载的问题。现在绝大多数应用的系统都是extractive的，这个方法比较简单但存在很多的问题，简单是因为只需要从原文中找出相对来说重要的句子来组成输出即可，系统只需要用模型来选择出信息量大的句子然后按照自然序组合起来就是摘要了。但是摘要的连贯性、一致性很难保证，比如遇到了句子中包含了代词，简单的连起来根本无法获知代词指的是什么，从而导致效果不佳。研究中随着deep learning技术在nlp中的深入，尤其是seq2seq+attention模型的“横行”，大家将abstractive式的摘要研究提高了一个level，并且提出了copy mechanism等机制来解决seq2seq模型中的OOV问题。

### 语料

这里的语料分为两种，一种是用来训练深度学习模型的大型语料，一种是用来参加评测的小型语料。

1、DUC

这个网站提供了文本摘要的比赛，2001-2007年在这个网站，2008年开始换到这个网站TAC。很官方的比赛，各大文本摘要系统都会在这里较量一番，一决高下。这里提供的数据集都是小型数据集，用来评测模型的。

2、Gigaword

该语料非常大，大概有950w篇新闻文章，数据集用headline来做summary，即输出文本，用first sentence来做input，即输入文本，属于单句摘要的数据集。

3、CNN/Daily Mail

该语料就是我们在机器阅读理解中用到的语料，该数据集属于多句摘要。

4、Large Scale Chinese Short Text Summarization Dataset（LCSTS）[LCSTS: A Large Scale Chinese Short Text Summarization Dataset](http://cn.arxiv.org/pdf/1506.05865)

这是一个中文短文本摘要数据集，数据采集自新浪微博。

### 1. Complex Attention Model

[Generating News Headlines with Recurrent Neural Networks](http://cn.arxiv.org/pdf/1512.01712)

模型中的attention weights是用encoder中每个词最后一层hidden layer的表示与当前decoder最新一个词最后一层hidden layer的表示做点乘，然后归一化来表示的。

### 2. Simple Attention Model

同样是[Generating News Headlines with Recurrent Neural Networks](http://cn.arxiv.org/pdf/1512.01712)

模型将encoder部分在每个词最后一层hidden layer的表示分为两块，一小块用来计算attention weights的，另一大块用来作为encoder的表示。这个模型将最后一层hidden layer细分了不同的作用。

### 3. Attention-Based Summarization(ABS)

[A Neural Attention Model for Abstractive Sentence Summarization](http://cn.arxiv.org/pdf/1509.00685.pdf)

这个模型用了三种不同的encoder，包括：Bag-of-Words Encoder、Convolutional Encoder和Attention-Based Encoder。Rush是HarvardNLP组的，这个组的特点是非常喜欢用CNN来做nlp的任务。这个模型中，让我们看到了不同的encoder，从非常简单的词袋模型到CNN，再到attention-based模型，而不是千篇一律的rnn、lstm和gru。而decoder部分用了一个非常简单的NNLM，就是Bengio[10]于2003年提出来的前馈神经网络语言模型，这一模型是后续神经网络语言模型研究的基石，也是后续对于word embedding的研究奠定了基础。可以说，这个模型用了最简单的encoder和decoder来做seq2seq，是一次非常不错的尝试。

### 4. ABS+

同样是[A Neural Attention Model for Abstractive Sentence Summarization](http://cn.arxiv.org/pdf/1509.00685.pdf)

Rush提出了一个纯数据驱动的模型ABS之后，又提出了一个abstractive与extractive融合的模型，在ABS模型的基础上增加了feature function，修改了score function，得到了这个效果更佳的ABS+模型。

### 5. Recurrent Attentive Summarizer(RAS)

[Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](http://nlp.seas.harvard.edu/papers/naacl16_summary.pdf)



### 6. big-words-lvt2k-1sent模型

[Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](http://cn.arxiv.org/pdf/1602.06023)

这个模型引入了large vocabulary trick(LVT)技术到文本摘要问题上。本方法中，每个mini batch中decoder的词汇表受制于encoder的词汇表，decoder词汇表中的词由一定数量的高频词构成。这个模型的思路重点解决的是由于decoder词汇表过大而造成softmax层的计算瓶颈。本模型非常适合解决文本摘要问题，因为摘要中的很多词都是来自于原文之中。

### 7. words-lvt2k-2sent-hieratt模型

同样是[Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](http://cn.arxiv.org/pdf/1602.06023)

文本摘要中经常遇到这样的问题，一些关键词出现很少但却很重要，由于模型基于word embedding，对低频词的处理并不友好，所以本文提出了一种decoder/pointer机制来解决这个问题。模型中decoder带有一个开关，如果开关状态是打开generator，则生成一个单词；如果是关闭，decoder则生成一个原文单词位置的指针，然后拷贝到摘要中。pointer机制在解决低频词时鲁棒性比较强，因为使用了encoder中低频词的隐藏层表示作为输入，是一个上下文相关的表示，而仅仅是一个词向量。这个pointer机制和后面有一篇中的copy机制思路非常类似。

### 8. feats-lvt2k-2sent-ptr模型

同样是[Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](http://cn.arxiv.org/pdf/1602.06023)

数据集中的原文一般都会很长，原文中的关键词和关键句子对于形成摘要都很重要，这个模型使用两个双向RNN来捕捉这两个层次的重要性，一个是word-level，一个是sentence-level，并且该模型在两个层次上都使用attention，权重如下：

### 9. COPYNET

[Incorporating Copying Mechanism in Sequence-to-Sequence Learning Training](http://cn.arxiv.org/pdf/1603.06393v2.pdf)

encoder采用了一个双向RNN模型，输出一个隐藏层表示的矩阵M作为decoder的输入。decoder部分与传统的Seq2Seq不同之处在于以下三部分：

+ 预测：在生成词时存在两种模式，一种是生成模式，一种是拷贝模式，生成模型是一个结合两种模式的概率模型。
+ 状态更新：用t-1时刻的预测出的词来更新t时刻的状态，COPYNET不仅仅词向量，而且使用M矩阵中特定位置的hidden state。
+ 读取M：COPYNET也会选择性地读取M矩阵，来获取混合了内容和位置的信息。

这个模型与第7个模型思想非常的类似，因为很好地处理了OOV的问题，所以结果都非常好。

### 10. MRT+NHG

[Neural Headline Generation with Minimum Risk Training](http://cn.arxiv.org/pdf/1604.01904.pdf)

这个模型的特别之处在于用了Minimum Risk Training训练数据，而不是传统的MLE（最大似然估计），将评价指标包含在优化目标内，更加直接地对评价指标做优化，得到了不错的结果。

### 效果

不管是中文数据集还是英文数据集上，最好的结果都是来自于模型10,并且该模型只是采用最普通的seq2seq+attention模型，都没有用到效果更好的copy机制或者pointer机制。

思考：

1. 为什么MRT那篇文章的结果会比其他各种各样的模型都要好呢？因为直接将ROUGE指标包含在了待优化的目标中，而不是与其他模型一样，采用传统的MLE来做，传统的目标评价的是你的生成质量如何，但与我们最终评价的指标ROUGE并无直接关系。所以说，换了一种优化目标，直接定位于评价指标上做优化，效果一定会很好。
2. OOV(out of vocabulary)的问题。因为文本摘要说到底，都是一个语言生成的问题，只要是涉及到生成的问题，必然会遇到OOV问题，因为不可能将所有词都放到词表中来计算概率，可行的方法是用选择topn个高频词来组成词表。文章[4]和[8]都采用了相似的思路，从input中拷贝原文到output中，而不仅仅是生成，**这里需要设置一个gate来决定这个词是copy来还是generate出来**。显然，增加了copy机制的模型会在很大程度上解决了OOV的问题，就会显著地提升评价结果。这种思路不仅仅在文摘问题上适用，在一切生成问题上都适用。
3. 关于评价指标的问题。一个评价指标是否科学直接影响了这个领域的发展水平，人工评价我们就不提了，只说自动评价。ROUGE指标在2003年就被Lin提出了[9]，13年过去了，仍然没有一个更加合适的评价体系来代替它。**ROUGE评价太过死板，只能评价出output和target之间的一些表面信息，并不涉及到语义层面上的东西**，是否可以提出一种更加高层次的评价体系，从语义这个层面来评价摘要的效果。其实技术上问题不大，因为计算两个文本序列之间的相似度有无数种解决方案，有监督、无监督、半监督等等等等。很期待有一种新的体系来评价摘要效果，相信新的评价体系一定会推动自动文摘领域的发展。
4. 关于数据集的问题。LCSTS数据集的构建给中文文本摘要的研究奠定了基础，将会很大程度地推动自动文摘在中文领域的发展。现在的互联网最不缺少的就是数据，大量的非结构化数据。但如何构建一个高质量的语料是一个难题，如何尽量避免用过多的人工手段来保证质量，如何用自动的方法来提升语料的质量都是难题。所以，如果能够提出一种全新的思路来构建自动文摘语料的话，将会非常有意义。
