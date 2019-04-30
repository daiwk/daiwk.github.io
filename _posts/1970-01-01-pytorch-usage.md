---
layout: post
category: "knowledge"
title: "pytorch常用函数"
tags: [pytorch, ]
---

目录

<!-- TOC -->

- [自动求导](#%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC)
- [torch常用操作](#torch%E5%B8%B8%E7%94%A8%E6%93%8D%E4%BD%9C)
- [torch.nn](#torchnn)
  - [torch.nn.functional](#torchnnfunctional)
  - [torch.nn.Conv2d](#torchnnconv2d)
  - [torch.nn.MaxPool2d](#torchnnmaxpool2d)
  - [torch.nn.Embedding](#torchnnembedding)
- [torch](#torch)
  - [torch.addmm](#torchaddmm)
  - [torch.mm](#torchmm)
  - [torch.bmm](#torchbmm)
  - [torch.unsqueeze](#torchunsqueeze)
- [torch.Tensor](#torchtensor)
  - [view](#view)
  - [masked_fill_](#maskedfill)

<!-- /TOC -->

## 自动求导

[https://blog.csdn.net/qjk19940101/article/details/79557204](https://blog.csdn.net/qjk19940101/article/details/79557204)

```python
#encoding=utf8
import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([2]), requires_grad=True)
y = x + 2 
z = y ** 2 + 3
print z
## 输出tensor([19.], grad_fn=<AddBackward>)

z.backward()
## z=(x+2)^2+3==>dz/dx=2(x+2)
print x.grad
## 输出tensor([8.])
```

参考[https://www.cnblogs.com/luckyscarlett/p/10552747.html](https://www.cnblogs.com/luckyscarlett/p/10552747.html)

## torch常用操作

[https://blog.csdn.net/xholes/article/details/81667211](https://blog.csdn.net/xholes/article/details/81667211)

## torch.nn

[https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/)

### torch.nn.functional

### torch.nn.Conv2d

```torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)```

+ in_channels(int) – 输入信号的通道
+ out_channels(int) – 卷积产生的通道
+ kerner_size(int or tuple) - 卷积核的尺寸
+ stride(int or tuple, optional) - 卷积步长
+ padding(int or tuple, optional) - 输入的每一条边补充0的层数
+ dilation(int or tuple, optional) – 卷积核元素之间的间距（空洞卷积时使用）
+ groups(int, optional) – 从输入通道到输出通道的阻塞连接数
+ bias(bool, optional) - 如果bias=True，添加偏置

正常的卷积如下：

+ 输入图片的shape是(height, width, in_channels)，
+ filter的shape是(height_f, width_f, in_channels)，**filter和输入的第三维必须相等**。
+ 对于一个filter而言，输出的图片是(height_o, width_o)，**注意，并没有第三维！！**
+ 所谓的outchannels就是filter的个数，所以输出是(height_o, width_o, out_channels)

卷积操作就是，将这个三维的filter（例如3x3x3）与输入图像的对应位置相乘，再将这27个数相加，得到的结果就是output的一个元素。

对于复杂的卷积，假设输入的尺度是`\((N, C_{in},H,W)\)`，输出尺度`\((N,C_out,H_out,W_out)\)`

`\[
out(N_i, C_{out_j})=bias(C_{out_j})+\sum^{C_{in}-1}_{k=0}weight(C_{out_j},k)\bigotimes input(N_i,k)
\]`

+ `\(\bigotimes\)`: 表示二维的相关系数计算 stride: 控制相关系数的计算步长 
+ dilation(空洞卷积): 用于控制内核点之间的距离，详细描述在[https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
+ groups: 控制输入和输出之间的连接： 
    + group=1，输出是所有的输入的卷积；
    + group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。

参数kernel_size，stride,padding，dilation也可以是一个int的数据，此时卷积height和width值相同;也可以是一个tuple数组，tuple的第一维度表示height的数值，tuple的第二维度表示width的数值

shape：

+ input: `\((N,C_{in},H_{in},W_{in})\)`
+ output: `\((N,C_{out},H_{out},W_{out})\)`
    + `\(H_{out}=floor((H_{in}+2padding[0]-dilation[0](kernerl\_size[0]-1)-1)/stride[0]+1)\)`
    + `\(W_{out}=floor((W_{in}+2padding[1]-dilation[1](kernerl\_size[1]-1)-1)/stride[1]+1)\)`

### torch.nn.MaxPool2d

```torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)```

pooling的基本思想，与卷积的运算基本一样，区别在于，filter和原图像做的不是卷积操作，而是对原图像的filter大小的区域做max/min之类的操作，而且filter并没有第三维，输入和输出的第三维是一样的。

针对pooling，并没有需要学习的参数。

如果输入的大小是(N,C,H,W)，那么输出的大小是(N,C,H_out,W_out)和池化窗口大小(kH,kW)的关系是：

`\[
out(N_i, C_j,k)=\max^{kH-1}_{m=0}\max^{kW-1}_{m=0}input(N_{i},C_j,stride[0]h+m,stride[1]w+n)
\]`

如果padding不是0，会在输入的每一边添加相应数目0

dilation用于控制内核点之间的距离，详细描述在[https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

参数kernel_size，stride, padding，dilation数据类型： 可以是一个int类型的数据，此时卷积height和width值相同; 也可以是一个tuple数组（包含来两个int类型的数据），第一个int数据表示height的数值，tuple的第二个int类型的数据表示width的数值

+ kernel_size(int or tuple) - max pooling的窗口大小
+ stride(int or tuple, optional) - max pooling的窗口移动的步长。**!!!!默认值是kernel_size!!!!!!**
+ padding(int or tuple, optional) - 输入的每一条边补充0的层数
+ dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
+ return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
+ ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

shape: 

+ 输入: `\((N,C,H_{in},W_{in})\)`
+ 输出: `\((N,C,H_{out},W_{out})\)`
    + `\(H_{out}=floor((H_{in} + 2padding[0] - dilation[0](kernel\_size[0] - 1) - 1)/stride[0] + 1\)`
    + `\(W_{out}=floor((W_{in} + 2padding[1] - dilation[1](kernel\_size[1] - 1) - 1)/stride[1] + 1\)`


### torch.nn.Embedding

[https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#sparse-layers](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#sparse-layers)。一个保存了固定字典和大小的简单查找表。这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。```torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)```。

+ num_embeddings：嵌入字典的大小。
+ embedding_dim：每个嵌入向量的大小。
+ padding_idx：如果提供的话，输出遇到此下标时用零填充。
+ max_norm：如果提供的话，会重新归一化词嵌入，使它们的范数小于提供的值。
+ norm_type：对于max_norm选项计算p范数时的p。
+ scale_grad_by_freq：如果提供的话，会根据字典中单词频率缩放梯度。

## torch

[https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/)

### torch.addmm

对矩阵mat1和mat2进行矩阵乘操作(用@表示)。矩阵mat加到最终结果。```out=(beta∗M)+(alpha∗mat1@mat2)```。```torch.addmm(beta=1, mat, alpha=1, mat1, mat2, out=None)```

### torch.mm

对矩阵mat1和mat2进行相乘。```torch.mm(mat1, mat2, out=None)```

### torch.bmm

对存储在两个批batch1和batch2内的矩阵进行批矩阵乘操作。```torch.bmm(batch1, batch2, out=None)```

用法：

```python
>>> batch1 = torch.randn(10, 3, 4)
>>> batch2 = torch.randn(10, 4, 5)
>>> res = torch.bmm(batch1, batch2)
>>> res.size()
torch.Size([10, 3, 5])
```

### torch.unsqueeze

返回一个新的张量，对输入的指定位置插入维度1。注意： 返回张量与输入张量共享内存，所以改变其中一个的内容会改变另一个。

用法：

```python
>>> max_len=10
>>> position = torch.arange(0, max_len).float().unsqueeze(1)
>>> position
tensor([[0.],
        [1.],
        [2.],
        [3.],
        [4.],
        [5.],
        [6.],
        [7.],
        [8.],
        [9.]])
>>> position = torch.arange(0, max_len).float().unsqueeze(0)
>>> position
tensor([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]])
>>>
```

## torch.Tensor

[https://pytorch-cn.readthedocs.io/zh/latest/package_references/Tensor/](https://pytorch-cn.readthedocs.io/zh/latest/package_references/Tensor/)

### view

```view(*args)```: 返回一个有相同数据但大小不同的tensor。 返回的tensor必须有与原tensor相同的数据和相同数目的元素，但可以有不同的大小。(类似reshape)

### masked_fill_

```masked_fill_(mask, value)```: 在mask值为1的位置处用value填充。mask的元素个数需和本tensor相同，但尺寸可以不同。
