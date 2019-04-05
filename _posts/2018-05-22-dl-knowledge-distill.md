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

因为梯度的量级(magnitude)被『软目标』缩放了`\(1/T^2\)`(没懂。。。下面好像有讲)，所以同时使用hard和soft target的时候，需要乘以`\(T^2\)`，这样可以保证即使T在实验的过程中改了，hard和soft targets的贡献程度相对不变。

### Matching logits是distillation的一个特例

交叉熵损失函数对小模型的logit，也就是`\(z_i\)`进行求导，得到`\(dC/dz_i\)`。假设大模型的logit是`\(v_i\)`，算出来的soft target的probability是`\(p_i\)`，那么：

