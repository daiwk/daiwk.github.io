---
layout: post
category: "ml"
title: "梯度下降优化算法"
tags: [梯度下降优化算法, momentum, NAG, Adagrad, Adadelta, RMSprop, Adam]
---

目录

<!-- TOC -->

- [梯度下降](#梯度下降)
- [momentum](#momentum)
- [NAG](#nag)
- [Adagrad](#adagrad)
- [Adadelta](#adadelta)
- [RMSprop](#rmsprop)
- [Adam](#adam)

<!-- /TOC -->

## 梯度下降

## momentum

如果在峡谷地区(某些方向较另一些方向上陡峭得多，常见于局部极值点)，SGD会在这些地方附近振荡，从而导致收敛速度慢。这种情况下，动量(Momentum)便可以解决。动量在参数更新项中加上一次更新量(即动量项)。

`\[
\\v_t=\gamma v_{t-1}+ \eta \triangledown _\theta\triangledown J(\theta)
\\\theta=\theta-v_t    
\]`

其中，`\(\eta\)`一开始初始化为0.5，后面变成0.9 [http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/)


## NAG

## Adagrad

## Adadelta

## RMSprop

## Adam

