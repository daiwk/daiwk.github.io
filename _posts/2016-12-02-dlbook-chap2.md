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
+ vectors：向量，一个1-D的数组。用小写斜体加粗表示`\(\boldsymbol{x}\in \mathbb{R}^n\)`：

`\[
\boldsymbol{x} = \begin{bmatrix}x_1
\\ x_2
\\ ...
\\ x_n
\end{bmatrix}
\]`

+ matrices：矩阵，一个2-D的数组。用大写斜体加粗表示`\(\boldsymbol{A}\in \mathbb{R}^{m\times n}\)`：
+ tensors：张量：0维==>标量；1维==>向量；2维==>矩阵；可以更多维`\(\boldsymbol{A_{i,j,k}}\in \mathbb{R}^{i\times j\times k}\)`：。