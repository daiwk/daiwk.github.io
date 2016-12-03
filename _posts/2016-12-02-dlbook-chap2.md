---
layout: post
category: "dlbook"
title: "deep learning book-第2章"
tags: [deep learning book,]
---

几个git链接：

+ [https://github.com/HFTrader/DeepLearningBook](https://github.com/HFTrader/DeepLearningBook)
+ [https://github.com/ExtremeMart/DeepLearningBook-ReadingNotes](https://github.com/ExtremeMart/DeepLearningBook-ReadingNotes)
+ [https://github.com/ExtremeMart/DeepLearningBook-CN](https://github.com/ExtremeMart/DeepLearningBook-CN)

# 2.1 Scalars, Vectors, Matrices and Tensors

+ Scalars：标量，一个单独的数字。用斜体小写字母表示`\(a,n,x\)`
+ vectors：向量，一个1-D的数组(**默认都是列向量！！**)。用小写斜体加粗表示`\(\boldsymbol{x}\in \mathbb{R}^n\)`：

`\[
\boldsymbol{x} = \begin{bmatrix}x_1
\\ x_2
\\ ...
\\ x_n
\end{bmatrix}
\]`

+ matrices：矩阵，一个2-D的数组。用大写斜体加粗表示`\(\boldsymbol{A}\in \mathbb{R}^{m\times n}\)`：
	+ 矩阵转置，简单理解就是将矩阵沿着主对角线做一次镜像：`\((\boldsymbol{A}^T)_{i,j}=\boldsymbol{A}_{j,i}\)`

![](../assets/deeplearningbook/chap2/matrix_transpose.jpg)

`\[
(\boldsymbol{A}\boldsymbol{B})^T=\boldsymbol{B}^T\boldsymbol{A}^T
\]`

+ tensors：张量：0维==>标量；1维==>向量；2维==>矩阵；可以更多维(用加粗大写非斜体表示)`\(\mathbf{A}\in \mathbb{R}^{i\times j\times k}\)`。

## 2.1.1 矩阵转置


