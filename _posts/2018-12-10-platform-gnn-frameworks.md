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
- [pgl](#pgl)

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

单机支持最大20亿边的图。

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

## pgl

paddle的graph learning

可以从这里搞些examples来试试：

[https://github.com/PaddlePaddle/PGL](https://github.com/PaddlePaddle/PGL)

安装不赘述了，还有官方文档：[https://pgl.readthedocs.io/en/latest/instruction.html](https://pgl.readthedocs.io/en/latest/instruction.html)

看看这个demo：

```python
import pgl
from pgl import graph  # import pgl module
import numpy as np

def build_graph():
    # define the number of nodes; we can use number to represent every node
    num_node = 10
    # add edges, we represent all edges as a list of tuple (src, dst)
    edge_list = [(2, 0), (2, 1), (3, 1),(4, 0), (5, 0),
             (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
             (7, 2), (7, 3), (8, 0), (9, 7)]

    # Each node can be represented by a d-dimensional feature vector, here for simple, the feature vectors are randomly generated.
    d = 16
    feature = np.random.randn(num_node, d).astype("float32")
    # each edge also can be represented by a feature vector
    edge_feature = np.random.randn(len(edge_list), d).astype("float32")

    # create a graph
    g = graph.Graph(num_nodes = num_node,
                    edges = edge_list,
                    node_feat = {'feature':feature},
                    edge_feat ={'edge_feature': edge_feature})

    return g

# create a graph object for saving graph data
g = build_graph()


print('There are %d nodes in the graph.'%g.num_nodes)
print('There are %d edges in the graph.'%g.num_edges)

# Out:
# There are 10 nodes in the graph.
# There are 14 edges in the graph.

import paddle.fluid as fluid

use_cuda = False
place = fluid.GPUPlace(0) if use_cuda else fluid.CPUPlace()

# use GraphWrapper as a container for graph data to construct a graph neural network
gw = pgl.graph_wrapper.GraphWrapper(name='graph',
                        place = place,
                        node_feat=g.node_feat_info())



# define GCN layer function
def gcn_layer(gw, feature, hidden_size, name, activation):
    # gw is a GraphWrapper；feature is the feature vectors of nodes

    # define message function
    def send_func(src_feat, dst_feat, edge_feat):
        # In this tutorial, we return the feature vector of the source node as message
        return src_feat['h']

    # define reduce function
    def recv_func(feat):
        # we sum the feature vector of the source node
        return fluid.layers.sequence_pool(feat, pool_type='sum')

    # trigger message to passing
    msg = gw.send(send_func, nfeat_list=[('h', feature)])
    # recv funciton receives message and trigger reduce funcition to handle message
    output = gw.recv(msg, recv_func)
    output = fluid.layers.fc(output,
                    size=hidden_size,
                    bias_attr=False,
                    act=activation,
                    name=name)
    return output

output = gcn_layer(gw, gw.node_feat['feature'],
                hidden_size=8, name='gcn_layer_1', activation='relu')
output = gcn_layer(gw, output, hidden_size=1,
                name='gcn_layer_2', activation=None)

y = [0,1,1,1,0,0,0,1,0,1]
label = np.array(y, dtype="float32")
label = np.expand_dims(label, -1)

# create a label layer as a container
node_label = fluid.layers.data("node_label", shape=[None, 1],
            dtype="float32", append_batch_size=False)

# using cross-entropy with sigmoid layer as the loss function
loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=output, label=node_label)

# calculate the mean loss
loss = fluid.layers.mean(loss)

# choose the Adam optimizer and set the learning rate to be 0.01
adam = fluid.optimizer.Adam(learning_rate=0.01)
adam.minimize(loss)

# create the executor
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
feed_dict = gw.to_feed(g) # gets graph data

for epoch in range(30):
    feed_dict['node_label'] = label

    train_loss = exe.run(fluid.default_main_program(),
        feed=feed_dict,
        fetch_list=[loss],
        return_numpy=True)
    print('Epoch %d | Loss: %f'%(epoch, train_loss[0]))
```
