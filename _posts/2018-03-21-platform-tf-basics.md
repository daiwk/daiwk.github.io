---
layout: post
category: "platform"
title: "tensorflow基础用法"
tags: [tensorflow基础, ]
---

目录

<!-- TOC -->

- [pandas](#pandas)
- [创建和操作张量](#创建和操作张量)
    - [矢量加法](#矢量加法)
    - [张量形状](#张量形状)
    - [广播](#广播)
    - [矩阵乘法](#矩阵乘法)
    - [张量变形](#张量变形)

<!-- /TOC -->

## pandas

## 创建和操作张量

### 矢量加法

```python
import tensorflow as tf
with tf.Graph().as_default():
  # Create a six-element vector (1-D tensor).
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

  # Create another six-element vector. Each element in the vector will be
  # initialized to 1. The first argument is the shape of the tensor (more
  # on shapes below).
  ones = tf.ones([6], dtype=tf.int32)

  # Add the two vectors. The resulting tensor is a six-element vector.
  just_beyond_primes = tf.add(primes, ones)

  # Create a session to run the default graph.
  with tf.Session() as sess:
    print just_beyond_primes.eval()
    #[ 3  4  6  8 12 14]
```

### 张量形状

```python
with tf.Graph().as_default():
  # A scalar (0-D tensor).
  scalar = tf.zeros([])

  # A vector with 3 elements.
  vector = tf.zeros([3])

  # A matrix with 2 rows and 3 columns.
  matrix = tf.zeros([2, 3])

  with tf.Session() as sess:
    print 'scalar has shape', scalar.get_shape(), 'and value:\n', scalar.eval()
    # 0.0
    print 'vector has shape', vector.get_shape(), 'and value:\n', vector.eval()
    # [0. 0. 0.]
    print 'matrix has shape', matrix.get_shape(), 'and value:\n', matrix.eval()
    #[[0. 0. 0.]
    # [0. 0. 0.]]
```

### 广播

利用广播，元素级运算中的较小数组会增大到与较大数组具有相同的形状。

* 如果指令需要大小为 `[6]` 的张量，则大小为 `[1]` 或 `[]` 的张量可以作为运算数。
* 如果指令需要大小为 `[4, 6]` 的张量，则以下任何大小的张量都可以作为运算数。
  * `[1, 6]`
  * `[6]`
  * `[]`
* 如果指令需要大小为 `[3, 5, 6]` 的张量，则以下任何大小的张量都可以作为运算数。
  * `[1, 5, 6]`
  * `[3, 1, 6]`
  * `[3, 5, 1]`
  * `[1, 1, 1]`
  * `[5, 6]`
  * `[1, 6]`
  * `[6]`
  * `[1]`
  * `[]`

当张量被广播时，从概念上来说，系统会复制其条目（出于性能考虑，**实际并不复制**。广播专为实现性能优化而设计）。

```python
with tf.Graph().as_default():
  # Create a six-element vector (1-D tensor).
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

  # Create a constant scalar with value 1.
  ones = tf.constant(1, dtype=tf.int32)

  # Add the two tensors. The resulting tensor is a six-element vector.
  just_beyond_primes = tf.add(primes, ones)

  with tf.Session() as sess:
    print just_beyond_primes.eval()
```

### 矩阵乘法

```python
with tf.Graph().as_default():
  # Create a matrix (2-d tensor) with 3 rows and 4 columns.
  x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]],
                  dtype=tf.int32)

  # Create a matrix with 4 rows and 2 columns.
  y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)

  # Multiply `x` by `y`. 
  # The resulting matrix will have 3 rows and 2 columns.
  matrix_multiply_result = tf.matmul(x, y)

  with tf.Session() as sess:
    print matrix_multiply_result.eval()
    # [[35 58]
    #  [35 33]
    #  [ 1 -4]]
```

### 张量变形

```python
with tf.Graph().as_default():
  # Create an 8x2 matrix (2-D tensor).
  matrix = tf.constant([[1,2], [3,4], [5,6], [7,8],
                        [9,10], [11,12], [13, 14], [15,16]], dtype=tf.int32)

  # Reshape the 8x2 matrix into a 2x8 matrix.
  reshaped_2x8_matrix = tf.reshape(matrix, (16,1))
  
  # Reshape the 8x2 matrix into a 4x4 matrix
  reshaped_4x4_matrix = tf.reshape(matrix, [4,4])

  with tf.Session() as sess:
    print "Original matrix (8x2):"
    print matrix.eval()
    print "Reshaped matrix (2x8):"
    print reshaped_2x8_matrix.eval()
    print "Reshaped matrix (4x4):"
    print reshaped_4x4_matrix.eval()
```

```python
with tf.Graph().as_default():
  # Create an 8x2 matrix (2-D tensor).
  matrix = tf.constant([[1,2], [3,4], [5,6], [7,8],
                        [9,10], [11,12], [13, 14], [15,16]], dtype=tf.int32)

  # Reshape the 8x2 matrix into a 3-D 2x2x4 tensor.
  reshaped_2x2x4_tensor = tf.reshape(matrix, [2,2,4])
  
  # Reshape the 8x2 matrix into a 1-D 16-element tensor.
  one_dimensional_vector = tf.reshape(matrix, [16])

  with tf.Session() as sess:
    print "Original matrix (8x2):"
    print matrix.eval()
    print "Reshaped 3-D tensor (2x2x4):"
    print reshaped_2x2x4_tensor.eval()
    print "1-D vector:"
    print one_dimensional_vector.eval()
```