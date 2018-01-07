---
layout: post
category: "dl"
title: "激活函数"
tags: [激活函数, activation function ]
---

目录

<!-- TOC -->

- [1. Step](#1-step)
- [2. Identity](#2-identity)
- [3. ReLU](#3-relu)
- [4. Sigmoid](#4-sigmoid)
- [5. Tanh](#5-tanh)
- [6. Leaky Relu](#6-leaky-relu)
- [7. PReLU](#7-prelu)
- [8. RReLU](#8-rrelu)
- [9. ELU](#9-elu)
- [10. SELU](#10-selu)
- [11. SReLU](#11-srelu)
- [12. Hard Sigmoid](#12-hard-sigmoid)
- [13. Hard Tanh](#13-hard-tanh)
- [14. LeCun Tanh](#14-lecun-tanh)
- [15. ArcTan](#15-arctan)
- [16. SoftSign](#16-softsign)
- [17. SoftPlus](#17-softplus)
- [18. Signum](#18-signum)
- [19. Bent Identity](#19-bent-identity)
- [20. Symmetrical Sigmoid](#20-symmetrical-sigmoid)
- [21. Log Log](#21-log-log)
- [22. Gaussian](#22-gaussian)
- [23. Absolute](#23-absolute)
- [24. Sinusoid](#24-sinusoid)
- [25. Cos](#25-cos)
- [26. Sinc](#26-sinc)

<!-- /TOC -->

参考[26种神经网络激活函数可视化](https://www.jiqizhixin.com/articles/2017-10-10-3)，原文[Visualising Activation Functions in Neural Networks](https://dashee87.github.io/data%20science/deep%20learning/visualising-activation-functions-in-neural-networks/)【可以在里面选择不同的激活函数，看图】

## 1. Step

## 2. Identity 

## 3. ReLU

## 4. Sigmoid

## 5. Tanh

## 6. Leaky Relu

## 7. PReLU

## 8. RReLU

## 9. ELU

指数线性单元（Exponential Linear Unit，ELU）也属于 ReLU 修正类激活函数的一员。和 PReLU 以及 RReLU 类似，为负值输入添加了一个非零输出。和其它修正类激活函数不同的是，它包括一个负指数项，从而防止静默神经元出现，导数收敛为零，从而提高学习效率。

`\[
f(x) = \begin{cases}\alpha(e^x - 1) & \text{for } x < 0\\x & \text{for } x \ge 0\end{cases}  
\]`

`\[
f'(x) = \begin{cases} \alpha e^{x} = f(x) + \alpha & \text{for } x < 0\\1 & \text{for } x \ge 0\end{cases}    
\]`

`\(\alpha = 0.7\)`：

<html>
<br/>
<img src='../assets/activations_elu_alpha0.7.png' style='max-height: 300px'/>
<br/>
</html>

`\(\alpha = 1\)`：

<html>
<br/>
<img src='../assets/activations_elu_alpha1.png' style='max-height: 300px'/>
<br/>
</html>

注：
tensorflow中的实现就是`\(\alpha = 1\)`的版本：

```c++
// Functor used by EluOp to do the computations.
template <typename Device, typename T>
struct Elu {
  // Computes Elu activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    // features.constant(?)
    activations.device(d) =
        (features < static_cast<T>(0))
            .select(features.exp() - features.constant(static_cast<T>(1)),
                    features);
  }
};

// Functor used by EluGradOp to do the computations.
template <typename Device, typename T>
struct EluGrad {
  // Computes EluGrad backprops.
  //
  // gradients: gradients backpropagated to the Elu op.
  // activations: outputs of the Elu op.
  // backprops: gradients to backpropagate to the Elu inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor activations,
                  typename TTypes<T>::Tensor backprops) {
    backprops.device(d) =
        (activations < static_cast<T>(0))
            .select((activations + static_cast<T>(1)) * gradients, gradients);
  }
};
```

## 10. SELU

参考
[引爆机器学习圈：「自归一化神经网络」提出新型激活函数SELU](https://zhuanlan.zhihu.com/p/27362891)
[加速网络收敛——BN、LN、WN与selu](http://skyhigh233.com/blog/2017/07/21/norm/)
[如何评价 Self-Normalizing Neural Networks 这篇论文?](https://www.zhihu.com/question/60910412)

paper: [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

其实就是ELU乘了个lambda，关键在于这个lambda是大于1的。以前relu，prelu，elu这些激活函数，都是在负半轴坡度平缓，这样在activation的方差过大的时候可以让它减小，防止了梯度爆炸，但是正半轴坡度简单的设成了1。而selu的正半轴大于1，在方差过小的的时候可以让它增大，同时防止了梯度消失。这样激活函数就有一个不动点，网络深了以后每一层的输出都是均值为0方差为1。

`\[
f(x) = \lambda \begin{cases}\alpha(e^x - 1) & \text{for } x < 0\\x & \text{for } x \ge 0 \end{cases}\\\text{ with } \lambda = 1.0507, \alpha = 1.67326
\]`

`\[
f'(x) = \begin{cases}\lambda\alpha e^{x} = (f(x) + \lambda \alpha) & \text{for } x < 0\\\lambda & \text{for } x \ge 0\end{cases}
\]`


tensorflow中的实现：
```c++
// Functor used by SeluOp to do the computations.
template <typename Device, typename T>
struct Selu {
  // Computes Selu activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    // features.constant(?)
    const auto scale = static_cast<T>(1.0507009873554804934193349852946);
    const auto scale_alpha = static_cast<T>(1.7580993408473768599402175208123);
    const auto one = static_cast<T>(1);
    const auto zero = static_cast<T>(0);
    activations.device(d) =
        (features < zero)
            .select(scale_alpha * (features.exp() - features.constant(one)),
                    scale * features);
  }
};

// Functor used by SeluGradOp to do the computations.
template <typename Device, typename T>
struct SeluGrad {
  // Computes SeluGrad backprops.
  //
  // gradients: gradients backpropagated to the Selu op.
  // activations: outputs of the Selu op.
  // backprops: gradients to backpropagate to the Selu inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor activations,
                  typename TTypes<T>::Tensor backprops) {
    const auto scale = static_cast<T>(1.0507009873554804934193349852946);
    const auto scale_alpha = static_cast<T>(1.7580993408473768599402175208123);
    backprops.device(d) =
        (activations < static_cast<T>(0)).select(
            gradients * (activations + scale_alpha), gradients * scale);
  }
};
```

## 11. SReLU


## 12. Hard Sigmoid

## 13. Hard Tanh

## 14. LeCun Tanh

## 15. ArcTan

## 16. SoftSign

Softsign 是 Tanh 激活函数的另一个替代选择。就像 Tanh 一样，Softsign 是反对称、去中心、可微分，并返回-1 和 1 之间的值。其更平坦的曲线与更慢的下降导数表明它可以更高效地学习。

`\[
f(x)=\frac{x}{1+|x|}, f(x) \in (-1,1)
\]`

`\[
f'(x)=\frac{1}{(1+|x|)^2}, f'(x) \in (0,1]
\]`

<html>
<br/>
<img src='../assets/activations_softsign.png' style='max-height: 300px'/>
<br/>
</html>

## 17. SoftPlus

## 18. Signum

## 19. Bent Identity

## 20. Symmetrical Sigmoid

## 21. Log Log

## 22. Gaussian

## 23. Absolute

## 24. Sinusoid

## 25. Cos

## 26. Sinc

