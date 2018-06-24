---
layout: post
category: "knowledge"
title: "tf常用函数"
tags: [tf, ]
---

目录

<!-- TOC -->

- [tf.reduce_*](#tfreduce_)
- [tf.nn.conv2d](#tfnnconv2d)
- [tf.nn.max_pool](#tfnnmax_pool)
- [tf.nn.relu](#tfnnrelu)
- [tf.contrib.layers.flatten](#tfcontriblayersflatten)
- [tf.contrib.layers.fully_connected](#tfcontriblayersfully_connected)

<!-- /TOC -->

## tf.reduce_*

实现代码在```tensorflow/python/ops/math_ops.py```

tensorflow中有一类在tensor的某一维度上求值的函数。

+ 求最大值tf.reduce_max()
+ 求平均值tf.reduce_mean()

参数：
+ input_tensor:待求值的tensor。
+ keepdims:是否保持其他维不变。（之前叫keep_dims）
+ axis:要对哪一维进行操作(之前叫reduction_indices)，只对这维求max/min，其他维删除。如果设置了keepdims=True，那么其他维的大小保持不变，要在[-rank(input_tensor), rank(input_tensor))范围内。

先看一个简单的例子：

```python
x=tf.constant([[1,4,3],[4,2,6]],dtype=tf.float32) # x.shape=(2, 3)
y = tf.reduce_max(x,axis=1,keepdims=True) 
# 2行3列，axis=1就在列维度操作，n列变成1列，即每一行求max，合到一列里
# 相当于只有第1维有值其他几维没东西了，第1维存的是其他几维的max
sess = tf.Session()
print x.shape
print sess.run(y)
print y.shape

y = tf.reduce_max(x,axis=0,keepdims=True) 
# 2行3列，axis=0就在行维度操作，n行变成1行，即每一列求max，合到一行里
# 相当于只有第0维有值其他几维没东西了，第0维存的是其他几维的max
sess = tf.Session()
print x.shape
print sess.run(y)
print y.shape
```

输出：

```python
(2, 3)
[[4.]
 [6.]]
(2, 1)

(2, 3)
[[4. 4. 6.]]
(1, 3)
```

再看个复杂一点的

```python
x=tf.constant([[[1,2,3],[4,5,6]],[[22,33,44],[55,66,77]]],dtype=tf.float32) # x.shape=(2, 2, 3)
y = tf.reduce_max(x,axis=0,keepdims=True)
sess = tf.Session()
print sess.run(y) 
print y.shape

y = tf.reduce_max(x,axis=1,keepdims=True)
sess = tf.Session()
print sess.run(y)
print y.shape

y = tf.reduce_max(x,axis=2,keepdims=True)
sess = tf.Session()
print sess.run(y)
print y.shape
```

输出：

```python
[[[22. 33. 44.]
  [55. 66. 77.]]]
(1, 2, 3)

[[[ 4.  5.  6.]]

 [[55. 66. 77.]]]
(2, 1, 3)

[[[ 3.]
  [ 6.]]

 [[44.]
  [77.]]]
(2, 2, 1)
```

## tf.nn.conv2d

**tf.nn.conv2d(X,W1, strides = [1,s,s,1], padding = 'SAME'):** given an input $X$ and a group of filters ```W1```, this function convolves ```W1```'s filters on X. The third input ([1,f,f,1]) represents the strides for each dimension of the input (m, n_H_prev, n_W_prev, n_C_prev). You can read the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)

实现代码在```tensorflow/python/ops/gen_nn_ops.py```中。

```python
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
```

输入的shape是```[batch, in_height, in_width, in_channels]```，即『NHWC』，一个kernel或filter的shape是```[filter_height, filter_width, in_channels, out_channels]```。这个函数实现如下功能：

+ 对filter进行flatten，变成一个shape是```[filter_height * filter_width * in_channels, output_channels]```的2D矩阵
+ 将input tensor的image patches进行extract，并组成一个shape是```[batch, out_height, out_width, filter_height * filter_width * in_channels]```的virtual tensor。
+ 对于每一个patch，right-multiplies the filter matrix and the image patch vector.

```python
output[b, i, j, k] =
    sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                    filter[di, dj, q, k]
```

必须满足```strides[0] = strides[3] = 1```，对于最common的case，也就是horizontal and vertices strides是一样的， ```strides = [1, stride, stride, 1]```

## tf.nn.max_pool

**tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME'):** given an input A, this function uses a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window. You can read the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool)

## tf.nn.relu

**tf.nn.relu(Z1):** computes the elementwise ReLU of Z1 (which can be any shape). You can read the full documentation [here.](https://www.tensorflow.org/api_docs/python/tf/nn/relu)

## tf.contrib.layers.flatten

**tf.contrib.layers.flatten(P)**: given an input P, this function flattens each example into a 1D vector it while maintaining the batch-size. It returns a flattened tensor with shape [batch_size, k]. You can read the full documentation [here.](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten)

## tf.contrib.layers.fully_connected

**tf.contrib.layers.fully_connected(F, num_outputs):** given a the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation [here.](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected)

