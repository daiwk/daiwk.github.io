---
layout: post
category: "knowledge"
title: "tf常用函数"
tags: [tf, ]
---

目录

<!-- TOC -->

- [tf.nn.conv2d](#tfnnconv2d)
- [tf.nn.max_pool](#tfnnmax_pool)
- [tf.nn.relu](#tfnnrelu)
- [tf.contrib.layers.flatten](#tfcontriblayersflatten)
- [tf.contrib.layers.fully_connected](#tfcontriblayersfully_connected)

<!-- /TOC -->

## tf.nn.conv2d

**tf.nn.conv2d(X,W1, strides = [1,s,s,1], padding = 'SAME'):** given an input $X$ and a group of filters ```W1```, this function convolves ```W1```'s filters on X. The third input ([1,f,f,1]) represents the strides for each dimension of the input (m, n_H_prev, n_W_prev, n_C_prev). You can read the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)

## tf.nn.max_pool

**tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME'):** given an input A, this function uses a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window. You can read the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool)

## tf.nn.relu

**tf.nn.relu(Z1):** computes the elementwise ReLU of Z1 (which can be any shape). You can read the full documentation [here.](https://www.tensorflow.org/api_docs/python/tf/nn/relu)

## tf.contrib.layers.flatten

**tf.contrib.layers.flatten(P)**: given an input P, this function flattens each example into a 1D vector it while maintaining the batch-size. It returns a flattened tensor with shape [batch_size, k]. You can read the full documentation [here.](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten)

## tf.contrib.layers.fully_connected

**tf.contrib.layers.fully_connected(F, num_outputs):** given a the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation [here.](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected)