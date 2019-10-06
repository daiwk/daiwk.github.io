---
layout: post
toc: true
category: "links"
title: "Useful Links"
tags: [useful links, mlflow, jaxnet, ludwig, knime, ]
---

目录：

<!-- TOC -->

- [**Basic Knowledges**](#basic-knowledges)
  - [Machine Learning](#machine-learning)
    - [**树模型**](#%e6%a0%91%e6%a8%a1%e5%9e%8b)
    - [**基础算法**](#%e5%9f%ba%e7%a1%80%e7%ae%97%e6%b3%95)
    - [**统计学习方法**](#%e7%bb%9f%e8%ae%a1%e5%ad%a6%e4%b9%a0%e6%96%b9%e6%b3%95)
  - [Deep Learning](#deep-learning)
    - [CNN](#cnn)
    - [RNN](#rnn)
    - [GAN](#gan)
    - [Reinforcement Learning](#reinforcement-learning)
    - [PNN(Progressive Neural Network)连续神经网络](#pnnprogressive-neural-network%e8%bf%9e%e7%bb%ad%e7%a5%9e%e7%bb%8f%e7%bd%91%e7%bb%9c)
    - [图卷积网络](#%e5%9b%be%e5%8d%b7%e7%a7%af%e7%bd%91%e7%bb%9c)
    - [copynet](#copynet)
- [**Useful Tools**](#useful-tools)
  - [Datasets](#datasets)
  - [pretrained models](#pretrained-models)
  - [Deep Learning Tools](#deep-learning-tools)
    - [mxnet](#mxnet)
    - [theano](#theano)
    - [torch](#torch)
    - [tensorflow](#tensorflow)
    - [docker](#docker)
    - [docker-images](#docker-images)
- [**Useful Courses && Speeches**](#useful-courses--speeches)
  - [Courses](#courses)
  - [Speeches](#speeches)
  - [经典学习资源](#%e7%bb%8f%e5%85%b8%e5%ad%a6%e4%b9%a0%e8%b5%84%e6%ba%90)
- [**Applications**](#applications)
  - [NLP](#nlp)
    - [分词](#%e5%88%86%e8%af%8d)
    - [Text Abstraction](#text-abstraction)
  - [Image Processing](#image-processing)
    - [image2txt](#image2txt)
  - [Collections](#collections)
    - [csdn深度学习代码专栏](#csdn%e6%b7%b1%e5%ba%a6%e5%ad%a6%e4%b9%a0%e4%bb%a3%e7%a0%81%e4%b8%93%e6%a0%8f)
    - [chiristopher olah的博客](#chiristopher-olah%e7%9a%84%e5%8d%9a%e5%ae%a2)
    - [激活函数系列](#%e6%bf%80%e6%b4%bb%e5%87%bd%e6%95%b0%e7%b3%bb%e5%88%97)
    - [梯度下降算法系列](#%e6%a2%af%e5%ba%a6%e4%b8%8b%e9%99%8d%e7%ae%97%e6%b3%95%e7%b3%bb%e5%88%97)

<!-- /TOC -->




[https://www.knime.com/](https://www.knime.com/)

Knime是一款出色的工具，可让你无需编写任何代码即可完成端到端的数据科学工作流程。

它甚至配备了一个拖放式界面，UI清晰，操作简单直观，可以说是懒人福音了。

操作起来非常简单，首先使用该工具进行数据收集和转换；完成后，你可以创建一个模型并将其可视化。在生产方面，你可以部署和管理数据科学项目。

最后，你可以通过使用Knime生成洞察来利用你的实现。

[https://uber.github.io/ludwig/getting_started/](https://uber.github.io/ludwig/getting_started/)

Uber Ludwig是另一款适合初学者的优秀工具。有了它，你可以快速测试和训练深度学习模型。用户可以选择启用懒人模式（拖拽界面），或者直接操作代码。

使用起来比Knime稍微复杂一点点。需要先加载CSV文件来训练数据。通过使用预先训练的模型，你可以预测输出目标。最后，你可以使用可用的可视化选项可视化你的数据。

如果你是编程的初学者，你还可以在Python中使用他们扩展的API和训练模型。

[https://github.com/mlflow/mlflow](https://github.com/mlflow/mlflow)

MLFlow让你可以解决端到端的机器学习生命周期问题。它有三个主要组件。

+ MLflow跟踪 - 通过记录和比较结果和参数来处理实验
+ MLflow项目 - 允许你将项目打包成其他成员的可重用表单
+ MLflow模型 - 帮助你在不同平台中部署和管理ML库

MLFlow的另一个惊人功能是它与库无关。这意味着你可以将其与其他机器学习库一起使用而不会出现任何兼容性问题。为了实现library-agonistic行为，它使用REST API和CLI。

动手学深度学习：[https://github.com/d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)

[https://zh.d2l.ai/](https://zh.d2l.ai/)

机器学习笔记：[https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/#tableofcontents](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/#tableofcontents)

jaxnet：[JAXnet：一行代码定义计算图，兼容三大主流框架，可GPU加速](https://mp.weixin.qq.com/s/mS40TE7_KmO6iAiDkrrTFg)

[https://paperswithcode.com/sota](https://paperswithcode.com/sota)

[https://www.jianshu.com/u/511ba5d71aef](https://www.jianshu.com/u/511ba5d71aef)

[https://www.cnblogs.com/pinard/](https://www.cnblogs.com/pinard/)

牛逼的网站：

工业级ml应用的templates：[http://predictionio.apache.org/](http://predictionio.apache.org/)，对应的代码[https://github.com/apache/predictionio](https://github.com/apache/predictionio)，例如推荐系统示例：[http://predictionio.apache.org/templates/recommendation/quickstart/](http://predictionio.apache.org/templates/recommendation/quickstart/)

自动上色，不过因为资金不足。。主网站关了。。[https://github.com/lllyasviel/style2paints](https://github.com/lllyasviel/style2paints)


[https://github.com/jwasham/coding-interview-university/blob/master/translations/README-cn.md](https://github.com/jwasham/coding-interview-university/blob/master/translations/README-cn.md)

# **Basic Knowledges**
 
## Machine Learning
 
### **树模型**
 
1. **Python & R的树模型**：[A Complete Tutorial on Tree Based Modeling from Scratch (in R & Python)](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python) 

### **基础算法**

1. **最小二乘的几何意义及投影矩阵**：[博客地址](http://mp.weixin.qq.com/s?__biz=MzA5ODUxOTA5Mg==&mid=2652550323&idx=1&sn=654ccf3d7cb12c68e5e7a2aa85907688&chksm=8b7e45e8bc09ccfe87b2d16a77205ae21a7ffcf2231d95a143cd06980e15997836425a4df284&scene=25#wechat_redirect)

### **统计学习方法**

1. **统计学习方法笔记**：[csdn博客](http://m.blog.csdn.net/article/details?id=8351337)

## Deep Learning 
 
### CNN
 
1. **人脸合成**：[《使用CNN进行人脸合成》](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2651987619&idx=2&sn=dfafcfe8956f4ca686532271cc1b0326&chksm=f1216a52c656e3446fe34187945a48e61ca49bdca187f5c56aab3d59829540e60645949856a4&mpshare=1&scene=1&srcid=1002uGDVOwvDY4eeWR5mzos9&pass_ticket=lD2bnuoAxxbEWgy8KxGVnWVzLL%2FeiSX9MsE68ZdaZQzVoXKXHlCJQ3sVCfTnR7MQ#rd)，代码地址：[https://github.com/zo7/facegen](https://github.com/zo7/facegen)
2. **Residual Net**：[《ICCV 2015 揭开152层神经网络的面纱》](https://mp.weixin.qq.com/s?__biz=MzAwMjM3MTc5OA==&mid=402120336&idx=1&sn=4bdc7abbfe47bc342c86129e1a18ff34&scene=18&mpshare=1&scene=1&srcid=1002wbRSWdlbb0r77vY5iWAo&pass_ticket=DoiMlYDlmCK%2FTS99n6JzBzzsHdN7QoyC81j%2BvUNHFkqqmuADrJsZlH0yXSTgpVEB#rd)
 
### RNN
 
1. **WILDML-RNN**：[part-1-4](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)；代码地址：[https://github.com/dennybritz/rnn-tutorial-gru-lstm](https://github.com/dennybritz/rnn-tutorial-gru-lstm)
2. **LSTM及其11种变种**：[《图解LSTM神经网络架构及其11种变体（附论文）》](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650719562&idx=1&sn=ad6693cdeaa18034ed1c53271f642ef7&chksm=871b0134b06c8822bf89781a81081c161eb82b06d0c20b655bd7b991202d363b6c233ef137ff&scene=0&pass_ticket=lD2bnuoAxxbEWgy8KxGVnWVzLL%2FeiSX9MsE68ZdaZQzVoXKXHlCJQ3sVCfTnR7MQ#rd)
2. **augmented-rnns**：google大脑的研究员在博客中讲述了Neural Turing Machine、Attentional Interfaces、Adaptive Computation Time和Neural Programmers四大部分。[英文原文](http://distill.pub/2016/augmented-rnns/)；[新智元翻译版](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2651986905&idx=4&sn=dcfdeb7c92826c0603569d5a86025536&chksm=f1216f28c656e63e309d7c92fd06a1c67ac96ebea2a8c6f90169cd944876fb367a8bf819b4f4&mpshare=1&scene=1&srcid=1002ho2GSC2PTnFhFUio3EYj&pass_ticket=DoiMlYDlmCK%2FTS99n6JzBzzsHdN7QoyC81j%2BvUNHFkqqmuADrJsZlH0yXSTgpVEB#rd)；gitbub博客代码：[https://github.com/distillpub/post--augmented-rnns](https://github.com/distillpub/post--augmented-rnns)
3. **漫谈4种RNN-decoder**：[博客地址](http://jacoxu.com/?p=1852)；github地址：[https://github.com/jacoxu/encoder_decoder](https://github.com/jacoxu/encoder_decoder)
4. 小合辑（从lstm到gru到双向多层到nmt到attention）:[https://daiwk.github.io/posts/nlp-nmt.html](https://daiwk.github.io/posts/nlp-nmt.html)
5. attention model集合：[https://daiwk.github.io/posts/dl-attention-models.html](https://daiwk.github.io/posts/dl-attention-models.html)
6. 只要attention的tensor2tensor：[https://daiwk.github.io/posts/platform-tensor-to-tensor.html](https://daiwk.github.io/posts/platform-tensor-to-tensor.html)
7. nested lstm: [https://daiwk.github.io/posts/dl-nested-lstm.html](https://daiwk.github.io/posts/dl-nested-lstm.html)
8. indRNN: [https://daiwk.github.io/posts/dl-indrnn.html](https://daiwk.github.io/posts/dl-indrnn.html)
9. colah的博客：翻译版之一：[https://www.leiphone.com/news/201701/UIlrDBnwiqoQUbqB.html](https://www.leiphone.com/news/201701/UIlrDBnwiqoQUbqB.html)，原文：[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

 
### GAN
 
1. **GAN简介**：[《Deep Learning Research Review Week 1: Generative Adversarial Nets》](https://adeshpande3.github.io/adeshpande3.github.io/Deep-Learning-Research-Review-Week-1-Generative-Adversarial-Nets)
2. **生成式对抗网络GAN研究进展系列笔记**：[http://blog.csdn.net/solomon1558/article/details/52537114](http://blog.csdn.net/solomon1558/article/details/52537114)
3. **cleverhans**：Ian Goodfellow等人在openai中开源了cleverhans，基于tf+keras+GPU：[https://github.com/openai/cleverhans](https://github.com/openai/cleverhans)
4. **GAN-zoo**：[https://github.com/hindupuravinash/the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)
 
### Reinforcement Learning

1. **WILDML-Deep Reinforcement Learning**：[http://www.wildml.com/2016/10/learning-reinforcement-learning/](http://www.wildml.com/2016/10/learning-reinforcement-learning/)
2. **强化学习概览**：NVIDIA 博客上 Tim Dettmers 所写的《Deep Learning in a Nutshell》系列文章的第四篇：[强化学习概览](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650719294&idx=1&sn=f1a01cd6710e6ea9629619cd3324d102&chksm=871b0040b06c895642ff961a6fe81f05c5e9776aff5da4845f2d3d874f88213863afd2059833&mpshare=1&scene=1&srcid=1002mDtDsEDixxCswQJOs2rH&pass_ticket=DoiMlYDlmCK%2FTS99n6JzBzzsHdN7QoyC81j%2BvUNHFkqqmuADrJsZlH0yXSTgpVEB#rd)

### PNN(Progressive Neural Network)连续神经网络
1. **PNN简介**：“我们想要从一个任务开始，在上面获得专家级别的表现，随后，我们迁移到另一个连续性的任务上，使用相同的神经网络来获得专家级别的表现，在这个过程中，神经网络不会忘掉此前学会的技巧，并可以在不同的任务间实现这些技巧的相互迁移。如果任务类似的话，我希望任务1中的技巧可以有效地迁移到任务4中。我想要实现的是，只要根据任务1进行训练，就能知道其中的技巧能否写入我的神经网络代码中，并可以迁移到下一个任务。”把单独的神经网络称为一个栏（Column），这些栏在神经网络的每一层旁边形成互连，并且， 我也会固定权重（模型的参数），这样我训练第二个栏的时候，我就知道如何使用栏1的特征，但是我不需要重新编写它们。[论文下载](https://arxiv.org/pdf/1606.04671.pdf)
[论文笔记](http://www.cnblogs.com/wangxiaocvpr/p/6002214.html)

### 图卷积网络
1. **Graph Convolutional Networks**：[http://tkipf.github.io/graph-convolutional-networks/](http://tkipf.github.io/graph-convolutional-networks/)

### copynet
1. **copynet**:[Incorporating Copying Mechanism in Sequence-to-Sequence Learning.pdf](../assets/Incorporating Copying Mechanism in Sequence-to-Sequence Learning.pdf) [github(基于theano，作者开源的)](https://github.com/MultiPath/CopyNet)
 
# **Useful Tools**
 
## Datasets
 
1. **Youtube-8m**：该数据集包含了 800 万个 YouTube 视频 URL（代表着 500,000 小时的视频）以及它们的视频层面的标签（video-level labels），这些标签来自一个多样化的包含 4800 个知识图谱实体（Knowledge Graph entity）的集合。相比于之前已有的视频数据集，这个数据集的规模和多样性都实现了显著的增长。比如说，我们所知的之前最大的视频数据集 Sports-1M 包含了大约 100 万段 YouTube 视频和 500 个体育领域的分类——YouTube-8M 在视频数量和分类数量上都差不多比它高一个数量级。论文：**《YouTube-8M: A Large-Scale Video Classification Benchmark》**
2. **Open Images(图片数据集，包含9百万标注图片)**:一个包含了900万图像URL的数据集，值得一提的是，这些图像全部都是标签数据，标签种类超过6000种。我们尽量让数据集变得实用：数据集中所使用的标签类型比拥有1000个分类的ImageNet数据集更加贴近实际生活。 [https://github.com/openimages/dataset](https://github.com/openimages/dataset)

## pretrained models

1. **大规模语言建模模型库(基于One Billion Word Benchmark)**：这个数据库含有大约 10 亿个单词，词汇有 80 万单词，大部分都是新闻数据。由于训练中句子是被打乱了的，模型可以不理会文本，集中句子层面的语言建模。在此基础上，作者在论文描述了一个模型，混合了字符CNN（character CNN）、大规模深度 LSTM，以及一个专门的 Softmanx 架构，最终得到的结果可以说是迄今最好的。[github](https://github.com/tensorflow/models/tree/master/lm_1b)
 
## Deep Learning Tools
 
### mxnet
1. **NNVM和tinyflow**：[《NNVM打造模块化深度学习系统》](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650719529&idx=3&sn=6992a6067c79349583762cb28eecda89&chksm=871b0157b06c8841587bdfb992c19290c8d66386a6f8accdf70998ce3f86b36330219c09672d&scene=0&pass_ticket=lD2bnuoAxxbEWgy8KxGVnWVzLL%2FeiSX9MsE68ZdaZQzVoXKXHlCJQ3sVCfTnR7MQ#rd)前端把计算表达成一个中间形式，通常我们称之为计算图，NNVM 则统一的对图做必要的操作和优化，然后再生成后端硬件代码。NNVM 是一个神经网络的比较高级的中间表示模块，它包含了图的表示以及执行无关的各种优化（例如内存分配，数据类型和形状的推导）。核心的是这两个github地址：[https://github.com/dmlc/nnvm](https://github.com/dmlc/nnvm)和[https://github.com/tqchen/tinyflow](https://github.com/tqchen/tinyflow)。
2. **tf-slim**：　今年早些时候，我们发布了图像分类模型 Inception V3 在 TensorFlow 上的运行案例。代码能够让用户使用同步梯度下降用 ImageNet 分类数据库训练模型。Inception V3 模型的基础是一个叫做 TF-Slim 的 TensorFlow 库，用户可以使用这个软件包定义、训练、评估 TensorFlow 模型。TF-Slim 库提供的常用抽象能使用户快速准确地定义模型，同时确保模型架构透明，超参数明确。有更多新的层（**比如 Atrous 卷积层和 Deconvolution**）、更多新的代价函数和评估指标（**如 mAP，IoU**），同时有很多pre-trained的模型（比如 **Inception、VGG、AlexNet、ResNet**）。	[https://github.com/tensorflow/models/tree/master/slim](https://github.com/tensorflow/models/tree/master/slim)

### theano
1. **bay area dl school's tutorial**：[https://github.com/daiwk/bayareadlschool-learning-theano](https://github.com/daiwk/bayareadlschool-learning-theano)

### torch
1. **bay area dl school's tutorial**：[https://github.com/daiwk/bayareadlschool-learning-torch](https://github.com/daiwk/bayareadlschool-learning-torch)

### tensorflow
1. **bay area dl school's tutorial**：[https://github.com/daiwk/bayareadlschool-learning-tensorflow](https://github.com/daiwk/bayareadlschool-learning-tensorflow)
2. **a tour of tensorflow**：[a tour of tensorflow](https://arxiv.org/pdf/1610.01178v1.pdf)

### docker
1. **nvidia-docker**：[https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

### docker-images

链接:**[http://pan.baidu.com/s/1kUU9znh](http://pan.baidu.com/s/1kUU9znh) 密码:yyfp**
 
# **Useful Courses && Speeches**
 
## Courses
1. **cs224d(nlp)**：课程链接：[http://cs224d.stanford.edu/syllabus.html](http://cs224d.stanford.edu/syllabus.html)；百度云课程pdf下载：[http://pan.baidu.com/s/1dFaA7PR](http://pan.baidu.com/s/1dFaA7PR)
2. **cs231n(cnn)**：课程链接：[http://cs231n.stanford.edu/](http://cs231n.stanford.edu/)
3. **Bay Area Deep Learning School 2016**：
课程安排（附课件下载链接）：
[http://www.bayareadlschool.org/schedule](http://www.bayareadlschool.org/schedule)

演讲视频：

day1: [youtube](https://www.youtube.com/watch?v=eyovmAtoUx0); [优酷](http://v.youku.com/v_show/id_XMTczNzYxNjg5Ng==.html)

day2：[youtube](https://www.youtube.com/watch?v=eyovmAtoUx0)；[优酷](http://v.youku.com/v_show/id_XMTczODc2ODE3Mg==.html)

daiwk整理: [deep reinforcement learning](http://pan.baidu.com/s/1mim7A2G)；[deep unsupervised learning](http://pan.baidu.com/s/1o8tVcue)

小结：[Yoshua Bengio压轴解读深度学习的基础和挑战](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650719442&idx=1&sn=ff9f8412f08dbb8e52cb1fdb748e5a4e&chksm=871b00acb06c89ba46582fe5c481a5bc93a69cca5e1eb9f0d86d03db99cb9db3b1c8fdaabde4&mpshare=1&scene=1&srcid=092660FGFS96T6aIXG9i1pI0&pass_ticket=DoiMlYDlmCK%2FTS99n6JzBzzsHdN7QoyC81j%2BvUNHFkqqmuADrJsZlH0yXSTgpVEB#rd)；[Andrej Karpathy 最新演讲计算机视觉深度学习技术与趋势](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2651987425&idx=1&sn=e5d4bb50352bf536d786bacb4cb16258&chksm=f1216910c656e006cdc2de9aa4cdf19b82f4dd79bb1ede4710f8876056381655e339a57b0bea&mpshare=1&scene=1&srcid=1002SLviFNa9PO6AXaJghl00&pass_ticket=DoiMlYDlmCK%2FTS99n6JzBzzsHdN7QoyC81j%2BvUNHFkqqmuADrJsZlH0yXSTgpVEB#rd)；

## Speeches
1. **google brain最近的7大研究项目**：[《谷歌大脑最近7大研究项目》](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2651987619&idx=1&sn=3f24b3384a9b10fce2001f4074b789ee&chksm=f1216a52c656e344b1f449f384fe0f41461c9e72992946a1b79c2d1d996eb4bef73e27e04f6c&scene=0&pass_ticket=lD2bnuoAxxbEWgy8KxGVnWVzLL%2FeiSX9MsE68ZdaZQzVoXKXHlCJQ3sVCfTnR7MQ#rd)

## 经典学习资源
1. **Michael Nielsen的neuralnetworksanddeeplearning**：[http://neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com)

# **Applications**
 
## NLP

### 分词
1. **THULAC.so**：THULAC（THU Lexical Analyzer for Chinese）由清华大学自然语言处理与社会人文计算实验室研制推出的一套中文词法分析工具包，具有中文分词和词性标注功能。[https://github.com/thunlp/THULAC.so](https://github.com/thunlp/THULAC.so)

### Text Abstraction
1. **Abstractive Text Summarization using Seq-to-Seq RNNs and Beyond**：来自IBM Watson，本文是一篇非常优秀的paper，在seq2seq+attention的基础上融合了很多的features、trick进来，提出了多组对比的模型，并且在多种不同类型的数据集上做了评测，都证明了本文模型更加出色。[张俊的分析](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247483777&idx=1&sn=d766a3dec1761bab4186cf89ce8a8723&mpshare=1&scene=1&srcid=0919dvXyKB1zjz6S543FH1Ib&pass_ticket=DoiMlYDlmCK%2FTS99n6JzBzzsHdN7QoyC81j%2BvUNHFkqqmuADrJsZlH0yXSTgpVEB#rd)

## Image Processing

### image2txt

1. **img2txt**：
[google博文链接](https://research.googleblog.com/2016/09/show-and-tell-image-captioning-open.html)；[论文链接](http://arxiv.org/abs/1609.06647)；[代码链接（tensorflow/img2txt）](https://github.com/tensorflow/models/tree/master/im2txt)

2. **karpathy**:[https://github.com/karpathy/neuraltalk](https://github.com/karpathy/neuraltalk)

[https://github.com/karpathy/neuraltalk2](https://github.com/karpathy/neuraltalk2)

3. **pretrained_resnet**：paddle的model_zoo中:[https://github.com/PaddlePaddle/Paddle/tree/develop/v1_api_demo/model_zoo/resnet](https://github.com/PaddlePaddle/Paddle/tree/develop/v1_api_demo/model_zoo/resnet)，用法:[http://doc.paddlepaddle.org/doc_cn/tutorials/imagenet_model/resnet_model_cn.html?highlight=zoo](http://doc.paddlepaddle.org/doc_cn/tutorials/imagenet_model/resnet_model_cn.html?highlight=zoo)

## Collections

### csdn深度学习代码专栏
[https://code.csdn.net/blog/41](https://code.csdn.net/blog/41)

### chiristopher olah的博客
[http://colah.github.io/](http://colah.github.io/)



### 激活函数系列

+ [激活函数](https://mp.weixin.qq.com/s?__biz=MzI1NTE4NTUwOQ==&mid=2650325236&idx=1&sn=7bd8510d59ddc14e5d4036f2acaeaf8d&mpshare=1&scene=1&srcid=1214qIBJrRhevScKXQQuqas4&pass_ticket=w2yCF%2F3Z2KTqyWW%2FUwkvnidRV3HF9ym5iEfJ%2BZ1dMObpcYUW3hQymA4BpY9W3gn4#rd)

### 梯度下降算法系列

+ [梯度下降优化算法](https://mp.weixin.qq.com/s?__biz=MzA5ODUxOTA5Mg==&mid=2652550294&idx=1&sn=820ddc89e1d1af35f14ccf645b963a76&chksm=8b7e45cdbc09ccdb985b3bbc22fbc0dcd013d53e9e9a6073d09d1b676b338af7bd8b7dd2a92d&mpshare=1&scene=1&srcid=0930LxixeTcq5wCcRStBTylE&pass_ticket=w2yCF%2F3Z2KTqyWW%2FUwkvnidRV3HF9ym5iEfJ%2BZ1dMObpcYUW3hQymA4BpY9W3gn4#rd)


