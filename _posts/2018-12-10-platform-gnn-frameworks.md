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
- [Angel 3.0](#angel-30)

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

## Angel 3.0

[腾讯开源全栈机器学习平台 Angel 3.0，支持三大类型图计算算法](https://mp.weixin.qq.com/s/bSpm72WIx061cFMFgkaMlw)

[https://github.com/Angel-ML/angel](https://github.com/Angel-ML/angel)

腾讯Angel在稀疏数据高维模型的训练上具有独特优势，擅长推荐模型和图网络模型相关领域。当前业界主流的大规模图计算系统主要有Facebook的Big Graph、Power graph、Data bricks的 Spark GraphX等，但这些系统并不都支持图挖掘、图表示学习、图神经网络的三大类型算法。

从性能上来看，Angel优于现有图计算系统，能够支持十亿级节点、千亿级边的传统图挖掘算法，以及百亿边的图神经网络算法需求。Angel可运行于多任务集群以及公有云环境，具备高效容错恢复机制，能够进行端到端的训练，新算法容易支持，同时，Angel能够支持图挖掘、图表示、图神经网络算法，具备图学习的能力。

Angel的PS是针对高维稀疏模型设计的, 而大图是非常高维、有多达十亿的节点，也是稀疏的, 因此PS架构也适合处理图数据。图算法有多种类型，如图挖掘算法、图表示学习、图神经网络。由于Angel的PS有自定义接口, 可以灵活地应对这几类算法，整个平台不需要改动，只要实现所需接口即可。关于可靠性问题，Angel从一开始就是针对共享集群、公有云环境设计的, 并与Spark的结合. Spark也具有很强的稳定性。易用性主要指与上下游是否完整配套。Spark On Angel可以与大数据处理结合，PyTorch On Angel可以跟深度学习结合，将把大数据计算、深度学习统一起来，用户不用借助第三方平台就能完成整个流程, 易用性好。

Angel可以运行在Yarn/Kubernetes环境上，它上面现在支持三类算法

+ 图挖掘: PageRank、Kcore、Closeness，共同好友、三角结构、社团发现、其他；
+ 图神经网络: GCN、GraphSage、DGI等神经网络算法；
+ 图表示学习: LINE、Node2Vec算法。

图算法比较多，先将这些算法分类，每一类采取不同的优化方式去实现和优化。

+ 第一类是三角结构类，数三角形。这类算法是暴力算法, 没有捷径可走。例如共同好友就是三角结构。基于三角结构可以实现一系列算法, 如Cluster Rank, Clustering coefficient, Ego Network.  
+ 第二类算法是连通分量，有WCC和SCC。这类算法核心的思想是要做图的折叠或者图的压缩。这类算法有一定的捷径可走，发现连通结点后，就可以进行合并,迭代时图在会不断变小，就可以加快迭代速度。
+ 第三类算法是节点的排序。比如PageRank、KCore、Closeness，这类算法的迭代轮数较多，可能好几百轮。有一些方法加速它，主要有两种，一种是有没有办法让它的迭代变得少一点，另一种是有没有办法让它每一轮迭代越来越快。
+ 第四类算法是图表示学习的算法, 也是没有捷径可走, 主要考虑一些图的划分策略。像GNN，也归为一类
+ 最后一类算法是图神经网络。图有很多节点，每个节点都有自己的特征。经过一层层的图卷积，每个节点上的特征就输出一个表示，再经过一层图卷积，又输出另外一层表示，不断的改变图每一个节点的表示，最后根据任务类型需求，对每一个节点的表示把它都加起来，再做softmax，对全图做分类。对任何两个节点，算他们俩俩相交，计算它们的概率，预测它们俩是不是有边。它的核心是一个图，一层卷积，两层卷积，然后输出。图神经网络的问题是图数据规模比较大，需要做深度学习。