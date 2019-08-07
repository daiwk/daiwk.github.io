---
layout: post
category: "platform"
title: "gnn frameworks"
tags: [gnn frameworks, dgl, pyg,]
---

目录

<!-- TOC -->

- [graph-nets(tf)](#graph-netstf)
- [DGL(mxnet+pytorch)](#dglmxnetpytorch)
- [PyG(pytorch)](#pygpytorch)
- [GraphVite](#graphvite)

<!-- /TOC -->

## graph-nets(tf)

[https://github.com/deepmind/graph_nets](https://github.com/deepmind/graph_nets)

..好像并没有多少功能。。。

## DGL(mxnet+pytorch)

参考[NYU、AWS联合推出：全新图神经网络框架DGL正式发布](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650753378&idx=2&sn=66cd0204e55439476745b751da8ecd61&chksm=871a8d1cb06d040a59581226fabbbfcdc2b70e8adcdfa33ee91662dbd99112cda94408858cc7&mpshare=1&scene=1&srcid=0309z4wr6KBh1FsmzZqQGJxC&pass_ticket=%2BP%2FN5cYeG852O%2FSNu1NE1SPUA8ubUIDrdxe7yapmhw5xuyc6UadTW4Gqxrxrq2TY#rd)

参考[专栏 \| 手把手教你用DGL框架进行批量图分类](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650756252&idx=3&sn=195a715f1e977a342b32137abe0a27d0&chksm=871a90e2b06d19f4c441859a50753466eda9971863ee3a69bcc198c06dc26469df603a4411c8&mpshare=1&scene=1&srcid=0309NNAbBhsEKRdQmH5T9ncP&pass_ticket=%2BP%2FN5cYeG852O%2FSNu1NE1SPUA8ubUIDrdxe7yapmhw5xuyc6UadTW4Gqxrxrq2TY#rd)

参考[性能提升19倍，DGL重大更新支持亿级规模图神经网络训练](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650761921&idx=3&sn=6ae05951d1f1dc119e3efabcdaee80a0&chksm=871aaebfb06d27a9f4c3cbde3fc0220959987ac41e7985be74de71b20614c47280dfeb2186d0&scene=0&xtrack=1&pass_ticket=OEoJxI2kFvfmi6pDQlY3W%2FGC2MeNgyiIRuMCWgKgSHf5DYmZLcpg4jkhV1VOz5EE#rd)

## PyG(pytorch)

参考[比DGL快14倍：PyTorch图神经网络库PyG上线了](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650758310&idx=1&sn=64655e53fddb1f9492b8b6a1fe3a382b&chksm=871a98d8b06d11ce8292442045b293b2819f5057b726e594a1896ea7c2ae0355949e9df7bd03&mpshare=1&scene=1&srcid=&pass_ticket=%2BP%2FN5cYeG852O%2FSNu1NE1SPUA8ubUIDrdxe7yapmhw5xuyc6UadTW4Gqxrxrq2TY#rd)


## GraphVite

[已开源！GraphVite 超高速图表示学习系统，1 分钟可学百万节点](https://mp.weixin.qq.com/s?__biz=MzU1NTUxNTM0Mg==&mid=2247491610&idx=1&sn=1188f6e0707031ee0a58ed10b89cb9d8&chksm=fbd18cbbcca605addf50058077ad5689bf3182b92e7302dacb6c55b2761bd518e33435848827&mpshare=1&scene=1&srcid=&sharer_sharetime=1565179278897&sharer_shareid=8e95986c8c4779e3cdf4e60b3c7aa752&pass_ticket=Kz97uXi0CH4ceADUC3ocCNkjZjy%2B0DTtVYOM7n%2FmWttTt5YKTC2DQT9lqCel7dDR#rd)

GraphVite 框架由两个部分组成，核心库和 Python wrapper。Python wrapper 可以为核心库中的类提供自动打包功能，并为应用程序和数据集提供了实现。

核心库用 C+11 和 CUDA 实现，并使用 pybind11 绑定到 python 中。它涵盖了 GraphVite 中所有与计算相关类的实现，例如图、求解器和优化器。所有这些成分都可以打包成类，这类似于 Python 接口。

在 C+实现中，Python 有一些不同之处。图和求解器由底层数据类型和嵌入向量长度实现。该设计支持 Python 接口中的动态数据类型，以及对最大化优化编译时（compile-time）。为了方便了对 GraphVite 的进一步开发，开发者还对 C+接口进行了高度抽象。通过连接核心接口，用户可以实现图形的深度学习例程，而无需关注调度细节。

+ include/base/实现基本数据结构
+ include/util/实现基本用途
+ include/core/实现优化器、图和求解器的核心接口
+ include/gpu/实现所有模型的前向和后向传播
+ include/instance/实现图和求解器的实例
+ include/bind.h 实现Python绑定
+ src/graphvite.cu 实例化所有Python类

[https://graphvite.io/](https://graphvite.io/)

[https://github.com/DeepGraphLearning/graphvite](https://github.com/DeepGraphLearning/graphvite)
