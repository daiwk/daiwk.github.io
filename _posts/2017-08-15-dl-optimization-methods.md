---
layout: post
category: "ml"
title: "梯度下降优化算法"
tags: [梯度下降优化算法, momentum, NAG, Adagrad, Adadelta, RMSprop, Adam]
---

目录

<!-- TOC -->

- [梯度下降](#%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D)
- [momentum](#momentum)
- [NAG](#nag)
- [Adagrad](#adagrad)
- [Adadelta](#adadelta)
- [RMSprop](#rmsprop)
- [Adam](#adam)

<!-- /TOC -->

## 梯度下降

参考[http://ruder.io/optimizing-gradient-descent/](http://ruder.io/optimizing-gradient-descent/)

部分算法有对应的中文版：[https://blog.csdn.net/u010089444/article/details/76725843](https://blog.csdn.net/u010089444/article/details/76725843)

优缺点参考：[https://blog.csdn.net/u012759136/article/details/52302426](https://blog.csdn.net/u012759136/article/details/52302426)

！！！假设模型的参数`\(\theta \in \mathbb{R}^{d}\)`

普通sgd有如下缺点：

+ 选择合适的learning rate比较困难
+ 对所有的参数更新使用同样的learning rate。对于稀疏数据或者特征，有时我们可能想**更新快**一些对于**不经常出现的特征**，对于**常出现**的特征**更新慢**一些，这时候SGD就不太能满足要求了
+ SGD容易收敛到**局部最优**，在某些情况下可能被困在**鞍点**【但是在合适的初始化和学习率设置下，鞍点的影响其实没这么大】

## momentum

如果在峡谷地区(某些方向较另一些方向上陡峭得多，常见于局部极值点)，SGD会在这些地方附近振荡，从而导致收敛速度慢。这种情况下，动量(Momentum)便可以解决。动量在参数更新项中加上一次更新量(即动量项)。

`\[
\begin{aligned} v_{t} &=\gamma v_{t-1}+\eta \nabla_{\theta} J(\theta) \\ \theta &=\theta-v_{t} \end{aligned}
\]`

其中，`\(\eta\)`一开始初始化为0.5，后面变成0.9。一般直接设成0.9好了。。[http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/)

简单地说，就是记录每次的权重更新量，这次的更新量就是**上次更新量乘上一个系数**，再加上学习率乘以梯度。

特点：

+ 下降**初期**，使用上一次参数更新，下降方向一致，乘上较大的`\(\gamma\)`能够进行很好的**加速**
+ 下降**中后期**时，在局部最小值来回震荡的时候，梯度接近0， `\(\gamma\)`使得**更新幅度变大**，可能可以跳出陷阱
+ **梯度改变方向**的时候，`\(\gamma\)`可以**减少更新**

## NAG

Nesterov accelerated gradient (NAG)

`\[
\begin{aligned} v_{t} &=\gamma v_{t-1}+\eta \nabla_{\theta} J\left(\theta-\gamma v_{t-1}\right) \\ \theta &=\theta-v_{t} \end{aligned}
\]`

momentum首先计算一个梯度(短的蓝色向量)，然后在加速更新梯度的方向进行一个大的跳跃(长的蓝色向量)，nesterov项首先在之前加速的梯度方向进行一个大的跳跃(棕色向量)，计算梯度然后进行校正(绿色梯向量)

<html>
<br/>
<img src='../assets/nag.png' style='max-height: 150px'/>
<br/>
</html>

## Adagrad

首先

`\[
g_{t, i}=\nabla_{\theta} J\left(\theta_{t, i}\right)
\]`

然后

`\[
\theta_{t+1, i}=\theta_{t, i}-\eta \cdot g_{t, i}
\]`

再然后

`\[
\theta_{t+1, i}=\theta_{t, i}-\frac{\eta}{\sqrt{G_{t, i i}+\epsilon}} \cdot g_{t, i}
\]`

其中，`\(G_{t} \in \mathbb{R}^{d \times d}\)`是一个对角矩阵，每个对角线位置`\(i,i\)`为对应参数`\(\theta_i\)`从第1轮到第t轮梯度的平方和。

综合起来看就是

`\[
\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{G_{t}+\epsilon}} \odot g_{t}
\]`

也就是学习率除以前t轮的梯度的平方和加上一个`\(\epsilon\)`再开根号。这个`\(\epsilon\)`一般设置成`\(1 e-8\)`。

特点：

+ 前期梯度小时，分母大，能放大梯度
+ 后期梯度大时，分母小，能约束梯度
+ 适合处理**稀疏梯度**

缺点：

+ 仍然依赖**人为设置的全局学习率**
+ 学习率设置得太大的话，会使得对梯度的调节太大
+ **中后期**，分母会越来越大，**梯度趋近0**，训练会**提前结束**

## Adadelta

有点长。。

首先

`\[
E\left[g^{2}\right]_{t}=\gamma E\left[g^{2}\right]_{t-1}+(1-\gamma) g_{t}^{2}
\]`

然后

`\[
\begin{array}{l}{\Delta \theta_{t}=-\eta \cdot g_{t, i}} \\ {\theta_{t+1}=\theta_{t}+\Delta \theta_{t}}\end{array}
\]`

再然后，adagrad是这样的：

`\[
\Delta \theta_{t}=-\frac{\eta}{\sqrt{G_{t}+\epsilon}} \odot g_{t}
\]`

而我们把那个平方和替换成平均：

`\[
\Delta \theta_{t}=-\frac{\eta}{\sqrt{E\left[g^{2}\right]_{t}+\epsilon}} g_{t}
\]`

其实就是RMS(root mean square)，也就是`\(R M S[g]_{t}=\sqrt{E\left[g^{2}\right]_{t}+\epsilon}\)`，代入上式可以得到：

`\[
\Delta \theta_{t}=-\frac{\eta}{R M S[g]_{t}} g_{t}
\]`

然后呢，定义一个参数平方的更新量`\(\Delta \theta _t^2\)`的指数平滑：

`\[
E\left[\Delta \theta^{2}\right]_{t}=\gamma E\left[\Delta \theta^{2}\right]_{t-1}+(1-\gamma) \Delta \theta_{t}^{2}
\]`

然后参数更新量`\(\Delta \theta _t\)`的RMSE就是：

`\[
R M S[\Delta \theta]_{t}=\sqrt{E\left[\Delta \theta^{2}\right]_{t}+\epsilon}
\]`

而`\(R M S[\Delta \theta]_{t}\)`是未知的，可以用直到上一个时间步的参数更新的RMS来近似。所以可以拿`\(R M S[\Delta \theta]_{t-1}\)`来替换学习率`\(\eta\)`，从而得到：

`\[
\begin{aligned} \Delta \theta_{t} &=-\frac{R M S[\Delta \theta]_{t-1}}{R M S[g]_{t}} g_{t} \\ \theta_{t+1} &=\theta_{t}+\Delta \theta_{t} \end{aligned}
\]`

特点：

+ 已经**不用依赖**于**全局学习率**了
+ 训练**初中期**，**加速**效果不错，很快
+ 训练**后期**，反复在**局部最小值附近抖动**

## RMSprop

`\[
\begin{aligned} E\left[g^{2}\right]_{t} &=0.9 E\left[g^{2}\right]_{t-1}+0.1 g_{t}^{2} \\ \theta_{t+1} &=\theta_{t}-\frac{\eta}{\sqrt{E\left[g^{2}\right]_{t}+\epsilon}} g_{t} \end{aligned}
\]`

就是在adagrad的基础上，分母那个平方和改成指数平滑，前一时间步0.9，当前时间步0.1，而且前一时间步依赖的是均值而不是平方和，与adadelta类似。hinton认为`\(\gamma\)`设成0.9，默认学习率`\(\eta\)`设0.01比较好。

特点：

+ 依赖于**全局学习率**
+ RMSprop算是Adagrad的一种发展，和Adadelta的变体，效果趋于二者之间
+ 适合处理**非平稳目标**
+ 对于**RNN效果很好**

## Adam

rmsprop可以看成是对二阶矩做了指数平滑，adam就是对一阶矩也做一个指数平滑。然后二阶和一阶平滑完后要scale一下。最后二阶的搞完做分母，一阶的搞完做分子。

两个的指数平滑如下：

`\[
\begin{aligned} m_{t} &=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\ v_{t} &=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2} \end{aligned}
\]`

然后：

`\[
\begin{aligned} \hat{m}_{t} &=\frac{m_{t}}{1-\beta_{1}^{t}} \\ \hat{v}_{t} &=\frac{v_{t}}{1-\beta_{2}^{t}} \end{aligned}
\]`

再然后：

`\[
\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}
\]`

默认值的话，`\(\beta_1=0.9\)`，`\(\beta_2=0.999\)`，`\(\epsilon=10^{-8}\)`

特点：

+ 结合了**Adagrad**善于处理**稀疏梯度**和**RMSprop**善于处理**非平稳目标**的优点
+ 对**内存**需求较**小**
+ 为不同的参数计算不同的自适应学习率
+ 也适用于大多**非凸优化**
+ 适用于**大数据集**和**高维空间**
