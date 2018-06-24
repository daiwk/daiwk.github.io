---
layout: post
category: "knowledge"
title: "numpy"
tags: [numpy, ]
---

目录

<!-- TOC -->

- [常用函数](#常用函数)
    - [np.shape](#npshape)
    - [np.multiply](#npmultiply)
    - [np.dot](#npdot)
    - [np.matmul](#npmatmul)

<!-- /TOC -->

## 常用函数

### np.shape

```python
W = np.random.randn(2,2,3,8)
np.shape(W) # (2,2,3,8)
```

### np.multiply

element-wise product

[https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html](https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html)

```python
>>> x1 = np.arange(9.0).reshape((3, 3))
>>> x1
array([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.],
       [ 6.,  7.,  8.]])
>>> x2 = np.arange(3.0)
>>> x2
array([ 0.,  1.,  2.])
>>> np.multiply(x1, x2)
array([[  0.,   1.,   4.],
       [  0.,   4.,  10.],
       [  0.,   7.,  16.]])
```

### np.dot

点乘

[https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html)

具体地：

+ If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
+ If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
+ If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
+ If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
+ If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:

```python
dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
```

对于两个2D-array而言，就是矩阵乘法（最好使用matmul）：

```python
>>> a = [[1, 0], [0, 1]]
>>> b = [[4, 1], [2, 2]]
>>> np.dot(a, b)
array([[4, 1],
       [2, 2]])
```

而

```python
>>> a = np.arange(3 * 4 * 5 * 6).reshape((3,4,5,6))
>>> b = np.arange(3 * 4 * 5 * 6)[::-1].reshape((5,4,6,3))
>>> np.dot(a, b)[2,3,2,1,2,2]
499128
>>> sum(a[2,3,2,:] * b[1,2,:,2])
499128
```

### np.matmul

矩阵乘法

[https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul)

+ If both arguments are 2-D they are multiplied like conventional matrices.
+ If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
+ If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
+ If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.

Multiplication by a scalar is not allowed, use ```*``` instead

matmul和dot的区别：

+ Multiplication by scalars is not allowed.
+ Stacks of matrices are broadcast together as if the matrices were elements.

```python
>>> a = [[1, 0], [0, 1]]
>>> b = [[4, 1], [2, 2]]
>>> np.matmul(a, b)
array([[4, 1],
       [2, 2]])
```