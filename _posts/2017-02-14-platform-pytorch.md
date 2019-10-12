---
layout: post
category: "platform"
title: "pytorch"
tags: [pytorch, 源码, pytorch 1.2, pytorch 1.3, ]
---


目录

<!-- TOC -->

- [pytorch源码解析](#pytorch%e6%ba%90%e7%a0%81%e8%a7%a3%e6%9e%90)
- [pytorch-lightning](#pytorch-lightning)
- [pytorch 1.2](#pytorch-12)
- [pytorch 1.3](#pytorch-13)
  - [PyTorch TorchScript](#pytorch-torchscript)
  - [TensorFlow Eager](#tensorflow-eager)

<!-- /TOC -->


中文文档：[https://www.pytorchtutorial.com/docs/](https://www.pytorchtutorial.com/docs/)


pytorch的多gpu使用tips：[https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)


## pytorch源码解析

[万字综述，核心开发者全面解读PyTorch内部机制](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650763179&idx=1&sn=c41e016ef58f4b4079bb70fbe05081f4&chksm=871aabd5b06d22c38550e6bdf2c645be073537d65e4a686c0345ca70f71ab68d8ff86f5ae35d&mpshare=1&scene=1&srcid=&pass_ticket=TloMdmvUbLd5jnKvVTzrccQhGuskwL6KQ0HhJLF56Nwtcb16%2BVvMA09bw32tFrjs#rd)

[PyTorch源码浅析：简介](https://www.52coding.com.cn/2019/05/05/PyTorch0/)

## pytorch-lightning

[https://github.com/williamFalcon/pytorch-lightning](https://github.com/williamFalcon/pytorch-lightning)

[基于PyTorch的「Keras」：除了核心逻辑通通都封装](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650767325&idx=1&sn=dc36e55d6201529e4fd3e984b8c618b1&chksm=871abba3b06d32b5ed875cb90060a3c5436f38464afc6ba9645efac32aaa8fd03ef99b5f1e0c&scene=0&xtrack=1&pass_ticket=Kz97uXi0CH4ceADUC3ocCNkjZjy%2B0DTtVYOM7n%2FmWttTt5YKTC2DQT9lqCel7dDR#rd)

## pytorch 1.2

参考[正式支持Transformer与TensorBoard，PyTorch 1.2新鲜出炉](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650767663&idx=3&sn=f4cf39ee5f0e0ac8d6e9c7e488237885&chksm=871a4551b06dcc47f751b413f331314ddcc9c142aaa79316bf87d523cec36b9d92f8a97e8a68&scene=0&xtrack=1&pass_ticket=mmBhl6hER5JU9q0KMKTTFnbwPDksdn18kk%2FlW9Ih3p2TCzi4%2BlfisKHhCysHq%2Bou#rd)

PyTorch 1.2 版本加入了标准的 nn.Transformer 模块。nn.Transformer 模块完全依赖注意机制描述输入和输出之间的全局依赖关系。nn.Transformer 模块的组件是单独设计的，以便于被分开使用。例如，没有更大的 nn.Transformer 的情况下，nn.TransformerEncoder 可被自身使用。

简单的输入```from torch.untils.tensorboard import SummaryWriter```就能启动 TensorBoard。一旦我们安装了 TensorBoard，PyTorch 的这项新 API 就会将模型和指标记录到 TensorBoard UI 的目录中，并进行可视化。它对所有基于 PyTorch、Caffe 2 的模型和张量都支持数量、图像、直方图、图和嵌入可视化。

## pytorch 1.3

[2019年度机器学习框架王者之战！PyTorch 1.3重磅发布，TensorFlow有未来吗？](https://mp.weixin.qq.com/s/n5iwQs_k8BLRnUrA8nlQhg)

[图灵奖得主力推：PyTorch 1.3 今天发布](https://mp.weixin.qq.com/s/LvPm4HuD5c09T4tQjumg_Q)

[PyTorch称霸学界，TensorFlow固守业界，ML框架之争将走向何方？](https://mp.weixin.qq.com/s/VlghKVslhM8gOLogAy5Igw)

### PyTorch TorchScript

PyTorch JIT是PyTorch的一个中间表征（IR），被称为TorchScript。TorchScript是PyTorch的「图」表征。你可以使用tracing或script模式把一个常规的PyTorch模型转换为TorchScript。

+ tracing接受一个**函数**和一个**输入**，记录下用该输入**执行的操作**，然后构建IR。虽然简单，但tracing也有其缺点。例如，它无法捕获还未执行的控制流。例如，如果执行了条件语句的true block，它就无法捕获false block。
+ Script模式接收一个**函数/类**，**重新解释Python代码**，然后直接输出TorchScript IR。这使得它可以支持任意代码，但它需要**重新解释 Python**。

一旦你的 PyTorch 模型在这个 IR 中，我们就得到了图模式的所有好处。我们可以在**没有Python依赖**的情况下**用C++部署PyTorch模型**，还可以优化该模型。

### TensorFlow Eager

在API层面上，TensorFlow的eager模式与PyTorch的eager模式基本相同，最初是由Chainer发明的。加入eager模式之后，TensorFlow就拥有了PyTorch eager模式的大部分优势（易用、可调试等）。

但这也给TensorFlow带来了相同的劣势。TensorFlow的eager模型**不能导出到非Python环境中**，**无法优化**，也**无法在移动端运行**。

这将TensorFlow置于与PyTorch相同的境地，它们的解决方式也基本相同——要么**trace**你的代码**（tf.function）**，要么**重新解释Python代码（Autograph）**。

因此，TensorFlow的eager模式也不是万能的。尽管你可以**用tf.function注释**将eager代码**转换为静态图**，但这并不是一个无缝过程（PyTorch的TorchScript也有类似问题）。tracing在根本上被限制了，**重新解释Python代码**本质上需要很大程度上**重写Python编译器**。当然，通过限制深度学习中用到的Python子集可以极大地简化这一范围。

在默认情况下启用eager模式时，TensorFlow强迫用户做出选择，要么为了易用性使用eager执行，这种做法**需要为了部署而重写**；**要么彻底不用eager执行**。PyTorch也面临相同的问题，但PyTorch可选择性加入的TorchScript似乎更加令人愉悦。
