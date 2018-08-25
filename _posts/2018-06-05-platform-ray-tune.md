---
layout: post
category: "platform"
title: "ray-tune"
tags: [ray, tune]
---

目录

<!-- TOC -->

- [简介](#%E7%AE%80%E4%BB%8B)
- [使用](#%E4%BD%BF%E7%94%A8)

<!-- /TOC -->

## 简介

[Tune: A Research Platform for Distributed Model Selection and Training](https://arxiv.org/pdf/1807.05118v1.pdf)
[https://ray.readthedocs.io/en/latest/tune.html](https://ray.readthedocs.io/en/latest/tune.html)

一个常见的例子涉及到模型的建立。 数据科学家要花费相当多的时间进行实验，其中许多涉及调整他们最爱的机器学习算法的参数。 随着深度学习和RL变得越来越流行，数据科学家将需要某种软件工具来进行高效的超参数调整和其他形式的实验和模拟。 RayTune是一个新的**深度学习和RL分布式超参数搜索框架**。 它建立在Ray之上，与RLlib紧密结合。 RayTune基于网格搜索，并使用early stopping的想法，包括**中位数停止**规则和**HyperBand**。

有越来越多的开放源代码软件工具可用于希望深入学习和RL的公司。我们处于经验时代，我们需要能够实现快速并行实验的工具，同时让我们能够利用流行的软件库，算法和组件。 Ray刚刚添加了两个库，让公司可以进行强化学习，并有效搜索神经网络架构的空间。

强化学习应用程序涉及多个组件，每个组件提供分布式计算的机会。 Ray RLlib采用了一种编程模型，可以轻松组合和重用组件，并利用多层次并行性和物理设备的并行性。在短期内，RISE实验室计划添加更多的RL算法，用于与在线服务集成的API，支持多智能体场景，以及一组扩展的优化策略。

## 使用

首先import

```python
import ray
import ray.tune as tune

ray.init()
```

然后对想要tune的函数，加一个reporter参数，并把metrics传给reporter:

```python
def train_func(config, reporter):  # add a reporter arg
     model = ( ... )
     optimizer = SGD(model.parameters(),
                     momentum=config["momentum"])
     dataset = ( ... )

     for idx, (data, target) in enumerate(dataset):
         accuracy = model.fit(data, target)
         reporter(mean_accuracy=accuracy) # report metrics
```

最后，设置搜索范围并执行：

```python
all_trials = tune.run_experiments({
    "my_experiment": {
        "run": train_func,
        "stop": {"mean_accuracy": 99},
        "config": {"momentum": tune.grid_search([0.1, 0.2])}
    }
})
```
