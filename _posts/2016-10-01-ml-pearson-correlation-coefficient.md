---
layout: post
category: "ml"
title: "pearson相关系数"
tags: [pearson相关系数, Pearson Correlation Coefficient]
---

目录

<!-- TOC -->

- [计算公式](#计算公式)
- [代码](#代码)

<!-- /TOC -->

参考：
[https://www.zhihu.com/question/19734616](https://www.zhihu.com/question/19734616)

## 计算公式
[http://blog.csdn.net/zhangjunjie789/article/details/51737366](http://blog.csdn.net/zhangjunjie789/article/details/51737366)

## 代码

```python
from math import sqrt

def multiply(a,b):
    #a,b两个列表的数据一一对应相乘之后求和
    sum_ab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sum_ab+=temp
    return sum_ab

def cal_pearson(x,y):
    n=len(x)
    #求x_list、y_list元素之和
    sum_x=sum(x)
    sum_y=sum(y)
    #求x_list、y_list元素乘积之和
    sum_xy=multiply(x,y)
    #求x_list、y_list的平方和
    sum_x2 = sum([pow(i,2) for i in x])
    sum_y2 = sum([pow(j,2) for j in y])
    molecular=sum_xy-(float(sum_x)*float(sum_y)/n)
    #计算Pearson相关系数，molecular为分子，denominator为分母
    denominator=sqrt((sum_x2-float(sum_x**2)/n)*(sum_y2-float(sum_y**2)/n))
    return molecular/denominator

```

皮尔逊相关系数的适用范围：

当两个变量的标准差都不为零时，相关系数才有定义，皮尔逊相关系数适用于： 
1. 两个变量之间是线性关系，都是连续数据。 
2. 两个变量的总体是正态分布，或接近正态的单峰分布。 
3. 两个变量的观测值是成对的，每对观测值之间相互独立。
