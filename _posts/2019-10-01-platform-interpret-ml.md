---
layout: post
category: "platform"
title: "InterpretML"
tags: [InterpretML, LIME, H2O, 可解释性, ]
---

目录

<!-- TOC -->

- [InterpretML](#interpretml)
- [其他库](#%e5%85%b6%e4%bb%96%e5%ba%93)
  - [LIME](#lime)
  - [H2O](#h2o)

<!-- /TOC -->

## InterpretML

[InterpretML: A Unified Framework for Machine Learning Interpretability](https://arxiv.org/abs/1909.09223v1)

[https://github.com/microsoft/interpret](https://github.com/microsoft/interpret)

InterpretML 是一个为实践者和研究者提供机器学习可解释性算法的开源 Python 软件包。InterpretML 能提供以下两种类型的可解释性：（1）明箱（glassbox），这是针对可解释性设计的机器学习模型（比如线性模型、规则列表、广义相加模型）；（2）黑箱（blackbox）可解释技术，用于解释已有的系统（比如部分依赖、LIME）。这个软件包可让实践者通过在一个统一的 API 下，借助内置的可扩展可视化平台，使用多种方法来轻松地比较可解释性算法。InterpretML 也包含了可解释 Boosting 机（Explanable Boosting Machine，EBM）的首个实现，这是一种强大的可解释明箱模型，可以做到与许多黑箱模型同等准确的性能。


## 其他库

参考[吐血整理！绝不能错过的24个顶级Python库](https://zhuanlan.zhihu.com/p/76112940)

### LIME

[“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf)

[https://github.com/marcotcr/lime](https://github.com/marcotcr/lime)

LIME是一种算法（库），可以解释任何分类器或回归量的预测。LIME是如何做到的呢？通过可解释的模型在局部不断接近预测值，这个模型解释器可用于生成任何分类算法的解释。

[在机器学习模型中建立信任（在Python中使用LIME）](https://www.analyticsvidhya.com/blog/2017/06/building-trust-in-machine-learning-models/)

### H2O

[https://github.com/h2oai/mli-resources](https://github.com/h2oai/mli-resources)

H2O的无人驾驶AI，提供简单的数据可视化技术，用于表示高度特征交互和非线性模型行为，通过可视化提供机器学习可解释性（MLI），说明建模结果和模型中特征的影响。

[机器学习可解释性](https://www.h2o.ai/wp-content/uploads/2018/01/Machine-Learning-Interpretability-MLI_datasheet_v4-1.pdf)
