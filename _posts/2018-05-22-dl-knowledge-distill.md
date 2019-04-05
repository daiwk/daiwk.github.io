---
layout: post
category: "dl"
title: "knowledge distill"
tags: [knowledge distill, ]
---

目录

<!-- TOC -->

- [introduction](#introduction)
- [distillation](#distillation)
    - [Matching logits是distillation的一个特例](#matching-logits是distillation的一个特例)
- [在MNIST上的初步实验](#在mnist上的初步实验)
- [在speech recognition上的实验](#在speech-recognition上的实验)
- [training ensembles of specialists on very big datasets](#training-ensembles-of-specialists-on-very-big-datasets)
    - [JFT dataset](#jft-dataset)
    - [specialist models](#specialist-models)
    - [assigning classes to specialists](#assigning-classes-to-specialists)
    - [performing inference with ensumbles of specialists](#performing-inference-with-ensumbles-of-specialists)
    - [结果](#结果)
- [soft targets as regularizers](#soft-targets-as-regularizers)
    - [使用soft targets以阻止specialists过拟合](#使用soft-targets以阻止specialists过拟合)
- [与mixtures of experts的关系](#与mixtures-of-experts的关系)

<!-- /TOC -->


[https://www.zhihu.com/question/50519680](https://www.zhihu.com/question/50519680)

原始paper：[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

参考[蒸馏神经网络(Distill the Knowledge in a Neural Network)](https://blog.csdn.net/zhongshaoyy/article/details/53582048)

## introduction

**核心思想**：一个复杂的网络结构模型是若干个单独模型组成的集合，或者是一些很强的约束条件下（比如dropout率很高）训练得到的一个很大的网络模型。一旦复杂网络模型训练完成，我们便可以用另一种训练方法：“蒸馏”，把复杂模型中的knowledge transfer到一个更易于部署的小模型中。

“蒸馏”的难点在于如何**缩减网络结构**但是把网络中的知识保留下来。知识就从输入向量到输出向量的一个learned mapping。做复杂网络的训练时，目标是将**正确答案的概率最大化**，但这引入了一个副作用：这种网络为**所有错误答案分配了概率**，即使这些概率非常小。而这些错误答案间也有的比别的大，例如一辆宝马车，错误答案货车的概率就比错误答案胡萝卜的概率要大得多，这也是模型泛化能力的体现。

将复杂模型转化为小模型时需要注意保留模型的泛化能力：

一种方法是利用由复杂模型产生的**分类概率**作为**软目标**来训练小模型。在transfer阶段，我们可以用同样的训练集或者是单独的transfer set。当复杂模型是由简单模型复合而成时，我们可以用各自的概率分布的算术平均或者几何平均作为软目标。当**软目标的熵值较高**时，相对硬目标，它每次训练**可以提供更多的信息和更小的梯度方差**，因此**小模型**可以**用更少的数据**和**更高的学习率**进行训练。

像MNIST这种任务，复杂模型可以给出很完美的结果，大部分信息分布在小概率的软目标中。比如一张2的图片被认为是3的概率为0.000001，被认为是7的概率是0.000000001，但对于cross entropy的损失函数的值来讲，就没什么区分性了，因为他们都接近0。

Caruana用logits（**softmax层的输入**）而不是softmax层的输出作为“软目标”。他们目标是是的**复杂模型的logits和小模型的logits的平方差最小**。

distillation：

+ 第一步，**提升**final softmax中的调节参数**T**，使得复杂模型**产生合适的『软目标』**。
+ 第二步，采用**同样的T**来**训练小模型**，让它去**匹配『软目标』**
+ 第三步，训练完成之后，**T变回1**

后面发现，匹配复杂模型的logits其实就是distillation的一个special case。

transfer set可以由无标签数据组成([Model Compression](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf))，也可以用原训练集。我们发现使用**原训练集**效果很好，特别是我们在**目标函数**中加了一项目之后，这一项会encourage小模型**预测真实目标**，同时**尽量匹配『软目标』**。要注意的是，小模型并不能完全无误的匹配“软目标”，而正确结果的犯错方向(erring in the direction of the correct answer)是有帮助的。


## distillation

softmax层公式如下：

`\[
q_i=\frac{\exp(z_i/T)}{\sum _j \exp(z_j/T)}
\]`

+ `\(z_i\)`：logit，也就是softmax层的输入
+ `\(q_i\)`：softmax层算出的分类概率
+ `\(T\)`：temperature，就是调节参数，一般设为1。**T越大，分类的概率分布越『软』**

复制粘贴一下上面提到的：

distillation：

+ 第一步，**提升**final softmax中的调节参数**T**，使得复杂模型**产生合适的『软目标』**。
+ 第二步，采用**同样的T**来**训练小模型**，让它去**匹配『软目标』**
+ 第三步，训练完成之后，**T变回1**

当transfer set中部分或者所有数据都有标签时，这种方式可以通过同时训练模型使得模型得到正确的标签来大大提升效果。

一种实现方法是用正确标签来修正『软目标』，但一种更好的方法是：对两个目标函数进行加权平均。

+ 第一个目标函数是两个模型的**『软目标』**的交叉熵，这个交叉熵用开始的那个**比较大的T**来计算。
+ 第二个目标函数是**正确标签**的交叉熵，这个交叉熵用**小模型softmax层的logits**来计算且**T等于1**。

发现当**第二个目标函数权重较低**时可以得到最好的结果。

因为梯度的量级(magnitude)被『软目标』缩放了`\(1/T^2\)`(下面有讲)，所以同时使用hard和soft target的时候，需要乘以`\(T^2\)`，这样可以保证即使T在实验的过程中改了，hard和soft targets的贡献程度相对不变。

### Matching logits是distillation的一个特例


那么，我们先看一下交叉熵的求导(参考[简单易懂的softmax交叉熵损失函数求导](https://blog.csdn.net/qian99/article/details/78046329))，假设`\(z_i\)`是logit，经过softmax后得到`\(a_i\)`，label是`\(y_i\)`，那么，由于n个类，只有一个类是1，其他都是0，所以`\(\sum_j y_j = 0\)`，所以：

`\[
\frac{\partial C}{\partial z_i}=\frac{\partial C}{\partial a_i}\frac{\partial a_i}{\partial z_i}=...=a_i\sum_j y_j-y_i=a_i-y_i
\]`

然后看回这个distill模型

+ 大模型的logit是`\(v_i\)`，算出来的soft target的probability是`\(p_i\)`，
+ 小模型的logit是`\(z_i\)`，算出来的soft target的probability是`\(q_i\)`

交叉熵损失函数对小模型的logit，也就是`\(z_i\)`进行求导（把`\(p_i\)`看成一个常量），得到的梯度`\(dC/dz_i\)`如下：

`\[
\frac{\partial C}{\partial z_i}=\frac{1}{T}(q_i-p_i)=\frac{1}{T}(\frac{e^{z_i/T}}{\sum_je^{z_j/T}}-\frac{e^{v_i/T}}{\sum_je^{v_j/T}})
\]`

然后，如果temperature T比logits的量级（magnitude）要大得多，那么，`\(z_i/T\)`趋向于0（是一个很小的数），`\(z_i<0\)`的时候是从左边趋向于0，`\(z_i\>0)`的时候是从右边趋向于0，所以，`\(e^{z_i/T}\approx e^0+z_i/T\)`。因此，可以如下方式近似：

`\[
\frac{\partial C}{\partial z_i}\approx \frac{1}{T}(\frac{1+z_i/T}{N+\sum _jz_j/T}-\frac{1+v_i/T}{N+\sum_jv_j/T})
\]`

假设对于每一个transfer case，都有logits的均值为0，那么就有`\(\sum_jz_j=\sum_jv_j=0\)`，所以上式可以简化为：

`\[
\frac{\partial C}{\partial z_i}\approx \frac{1}{T}(\frac{1+z_i/T}{N}-\frac{1+v_i/T}{N})=\frac{1}{NT^2}(z_i-v_i)
\]`

所以，如果temperature T很高，如果对于每一个transfer case，都有logits的均值为0，那么distillation就等价于最小化`\(1/2(z_i-v_i)^2\)`，也就是Caruana提出的使得复杂模型的logits和小模型的logits的平方差最小。

而对于比较低的temperature T来讲，distillation对那些比平均值negative很多的logits的matching，会给予更少的关注。因为这样的logits在大模型的损失函数中几乎是unconstrained，也就是noisy的，所以这是potentially advantageous的。另一方面，这些很negative的logits可能可以传递大模型学到的知识中的很有用的信息。上面的这些效果哪个起了决定性作用其实是一个empirical(经验主义) question。当distilled model比大模型小太多，以至于无法捕捉到大模型的所有知识时，intermediate（中间的）的temperature效果最好，强烈建议把large negative logits直接忽略掉是很有用的。

## 在MNIST上的初步实验

## 在speech recognition上的实验

## training ensembles of specialists on very big datasets

### JFT dataset

### specialist models

### assigning classes to specialists

### performing inference with ensumbles of specialists

### 结果

## soft targets as regularizers

### 使用soft targets以阻止specialists过拟合

## 与mixtures of experts的关系


