---
layout: post
category: "ml"
title: "svm理解"
tags: [svm理解, ]
---

参考 
**支持向量机通俗导论（理解SVM的三层境界）**：[http://blog.csdn.net/v_july_v/article/details/7624837](http://blog.csdn.net/v_july_v/article/details/7624837)

## 引言

logistic regression: 给sigmoid之前的函数值`\(\theta^Tx\)`是负无穷到正无穷，sigmoid之后的值相当于取y=1时的概率值，大于0.5就视为1的类。

