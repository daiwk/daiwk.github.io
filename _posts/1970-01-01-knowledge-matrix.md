---
layout: post
category: "knowledge"
title: "机器学习中的矩阵、向量求导"
tags: [机器学习中的矩阵、向量求导, ]
---

目录

<!-- TOC -->

- [常用梯度](#%E5%B8%B8%E7%94%A8%E6%A2%AF%E5%BA%A6)
  - [交叉熵](#%E4%BA%A4%E5%8F%89%E7%86%B5)

<!-- /TOC -->

参考[机器之心最干的文章：机器学习中的矩阵、向量求导](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650737481&idx=1&sn=10e82e52991eb87170e22109857c3dec&chksm=871acf37b06d4621aabc409f95c72f935670c8595f274d62bf5283fdf275f052f29b73c27ab0&mpshare=1&scene=1&srcid=0207pDzvA7Kk5C3JAPj19P1a&pass_ticket=SeHiwrhprfhEeBDsE1XoKKgiqXKD0hs5Oyunmf09XE%2BrWYKA98pvhGxAVGX75FF1#rd)

[pdf地址](https://daiwk.github.io/assets/matrix+vector+derivatives+for+machine+learning.pdf)

## 常用梯度

### 交叉熵

```python
    mf1_v = [0.2, 0.2, 0.3]
    mf2_v = [0.1, 0.2, 0.3]
    lr1_v = [0.1]#, 0.2, 0.3]
    lr2_v = [0.1]#, 0.2, 0.3]
    bias_v = [0.1]#, 0.1, 0.2]
    mf1 = Variable(torch.Tensor(mf1_v), requires_grad=True)
    mf2 = Variable(torch.Tensor(mf2_v), requires_grad=True)
    lr1 = Variable(torch.Tensor(lr1_v), requires_grad=False)
    lr2 = Variable(torch.Tensor(lr2_v), requires_grad=False)
    bias = Variable(torch.Tensor(bias_v), requires_grad=False)

    ss = torch.sigmoid(mf1.dot(mf2) + lr1 + lr2 + bias) 
    ss.backward()
    print "torch.sigmoid(mf1.dot(mf2)+ lr1 + lr2) GRAD:", mf1.grad
    print "Check torch.sigmoid(mf1.dot(mf2) + lr1 + lr2) GRAD:", ss * (1 - ss) * mf2
    mf1.grad.data.zero_()
    print mf1.grad
```

然后看看正例和负例的loss的梯度：

```python
    ## z = wx+b
    ## cross-entropy loss = -(y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z)))
    ## cross-entropy loss's w gradient: x(sigmoid(z)-y)
    ## cross-entropy loss's b gradient: (sigmoid(z)-y)
    ss = torch.sigmoid(mf1.dot(mf2) + lr1 + lr2 + bias) 

    positive_loss = -torch.log(ss)
    positive_loss.backward()
    print "positive loss mf1 GRAD:", mf1.grad
    print "Check positive loss mf1 GRAD:", mf2 * (ss - 1) # y=1
    print "positive loss mf2 GRAD:", mf2.grad
    print "Check positive loss mf2 GRAD:", mf1 * (ss - 1) # y=1
    mf1.grad.data.zero_()
    mf2.grad.data.zero_()
    print mf1.grad
    print mf2.grad

    ss2 = torch.sigmoid(mf1.dot(mf2) + lr1 + lr2 + bias)
    negative_loss = -torch.log(1-ss2)
    negative_loss.backward()
    print "negative loss mf1 GRAD:", mf1.grad
    print "Check negative loss mf1 GRAD:", mf2 * (ss) # y=0
    print "negative loss mf2 GRAD:", mf2.grad
    print "Check negative loss mf2 GRAD:", mf1 * (ss) # y=0
    mf1.grad.data.zero_()
    mf2.grad.data.zero_()
    print mf1.grad
    print mf2.grad
```
