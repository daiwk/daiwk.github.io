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

有个中文版：[https://blog.csdn.net/u010089444/article/details/76725843](https://blog.csdn.net/u010089444/article/details/76725843)

！！！假设模型的参数`\(\theta \in \mathbb{R}^{d}\)`

## momentum

如果在峡谷地区(某些方向较另一些方向上陡峭得多，常见于局部极值点)，SGD会在这些地方附近振荡，从而导致收敛速度慢。这种情况下，动量(Momentum)便可以解决。动量在参数更新项中加上一次更新量(即动量项)。

`\[
\begin{aligned} v_{t} &=\gamma v_{t-1}+\eta \nabla_{\theta} J(\theta) \\ \theta &=\theta-v_{t} \end{aligned}
\]`

其中，`\(\eta\)`一开始初始化为0.5，后面变成0.9。一般直接设成0.9好了。。[http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/)

简单地说，就是记录每次的权重更新量，这次的更新量就是**上次更新量乘上一个系数**，再加上学习率乘以梯度。

## NAG

Nesterov accelerated gradient (NAG)

`\[
\begin{aligned} v_{t} &=\gamma v_{t-1}+\eta \nabla_{\theta} J\left(\theta-\gamma v_{t-1}\right) \\ \theta &=\theta-v_{t} \end{aligned}
\]`

## Adagrad

`\[
\begin{aligned} v_{t} &=\gamma v_{t-1}+\eta \nabla_{\theta} J\left(\theta-\gamma v_{t-1}\right) \\ \theta &=\theta-v_{t} \end{aligned}
\]`

然后

`\[
g_{t, i}=\nabla_{\theta} J\left(\theta_{t, i}\right)
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

## Adadelta

有点长。。

## RMSprop

`\[
\begin{aligned} E\left[g^{2}\right]_{t} &=0.9 E\left[g^{2}\right]_{t-1}+0.1 g_{t}^{2} \\ \theta_{t+1} &=\theta_{t}-\frac{\eta}{\sqrt{E\left[g^{2}\right]_{t}+\epsilon}} g_{t} \end{aligned}
\]`

就是在adagrad的基础上，分母那个平方和改成指数平滑，前一时间步0.9，当前时间步0.1。hinton认为`\(\gamma\)`设成0.9，不过默认值设0.01比较好。

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
