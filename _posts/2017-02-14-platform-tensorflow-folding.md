---
layout: post
category: "platform"
title: "tensorflow-folding/eager execution"
tags: [tensorflow, tensorflow-folding, eager execution]
---

目录

<!-- TOC -->

- [tensorflow fold](#tensorflow-fold)
- [eager execution](#eager-execution)

<!-- /TOC -->


## tensorflow fold

[知乎专栏：以静制动的TensorFlow Fold](https://zhuanlan.zhihu.com/p/25216368?utm_medium=social)


## eager execution

[终于！TensorFlow引入了动态图机制Eager Execution](https://blog.csdn.net/uwr44uouqcnsuqb60zk2/article/details/78431019)

Eager Execution 的优点如下：

+ 快速调试即刻的运行错误并通过 Python 工具进行整合
+ 借助易于使用的 Python 控制流支持动态模型
+ 为自定义和高阶梯度提供强大支持
+ 适用于几乎所有可用的 TensorFlow 运算

当你启动 Eager Execution 时，运算会即刻执行，无需 Session.run() 就可以把它们的值返回到 Python。比如，要想使两个矩阵相乘：

```python
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
x = [[2.]]
m = tf.matmul(x, x)
```

然后就可以直接

```python
print(m)
```
