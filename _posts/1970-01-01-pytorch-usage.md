---
layout: post
category: "knowledge"
title: "pytorch常用函数"
tags: [pytorch, ]
---

目录



## torch.nn

[https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/)

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
