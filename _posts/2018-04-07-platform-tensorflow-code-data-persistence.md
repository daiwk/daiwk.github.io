---
layout: post
category: "platform"
title: "tensorflow代码——数据持久化"
tags: [tensorflow代码, 数据持久化]
---

目录

<!-- TOC -->

- [MetaGraphDef](#metagraphdef)
    - [meta_info_def](#metainfodef)
    - [graph_def](#graphdef)
    - [saver_def](#saverdef)
    - [collection_def](#collectiondef)

<!-- /TOC -->

参考：《TensorFlow实战Google深度学习框架（第2版）》第5章

## MetaGraphDef

tf通过**元图（MetaGraph）**记录计算图中**节点的信息**及运行计算图中**节点所需的元数据**。

在```tensorflow/core/protobuf/meta_graph.proto```中定义了：

```c++
message MetaGraphDef {
  // Meta information regarding the graph to be exported.  To be used by users
  // of this protocol buffer to encode information regarding their meta graph.
  message MetaInfoDef {
    // User specified Version string. Can be the name of the model and revision,
    // steps this model has been trained to, etc.
    string meta_graph_version = 1;

    // A copy of the OpDefs used by the producer of this graph_def.
    // Descriptions and Ops not used in graph_def are stripped out.
    OpList stripped_op_list = 2;

    // A serialized protobuf. Can be the time this meta graph is created, or
    // modified, or name of the model.
    google.protobuf.Any any_info = 3;

    // User supplied tag(s) on the meta_graph and included graph_def.
    //
    // MetaGraphDefs should be tagged with their capabilities or use-cases.
    // Examples: "train", "serve", "gpu", "tpu", etc.
    // These tags enable loaders to access the MetaGraph(s) appropriate for a
    // specific use-case or runtime environment.
    repeated string tags = 4;

    // The __version__ string of the tensorflow build used to write this graph.
    // This will be populated by the framework, which will overwrite any user
    // supplied value.
    string tensorflow_version = 5;

    // The __git_version__ string of the tensorflow build used to write this
    // graph. This will be populated by the framework, which will overwrite any
    // user supplied value.
    string tensorflow_git_version = 6;

    // A flag to denote whether default-valued attrs have been stripped from
    // the nodes in this graph_def.
    bool stripped_default_attrs = 7;
  }
  MetaInfoDef meta_info_def = 1;

  // GraphDef.
  GraphDef graph_def = 2;

  // SaverDef.
  SaverDef saver_def = 3;

  // collection_def: Map from collection name to collections.
  // See CollectionDef section for details.
  map<string, CollectionDef> collection_def = 4;

  // signature_def: Map from user supplied key for a signature to a single
  // SignatureDef.
  map<string, SignatureDef> signature_def = 5;

  // Asset file def to be used with the defined graph.
  repeated AssetFileDef asset_file_def = 6;
}
```

保存MetaGraphDef的文件默认以.meta结尾，是二进制文件。tf有export_meta_graph函数，可以以json格式导出MetaGraphDef：

```python
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1], name="v1"))
v2 = tf.Variable(tf.constant(13.8, shape=[1], name="v2"))

result = v1 + v2

saver = tf.train.Saver()

ckpt_json_path = "./demo/model/model.ckpt.meta.json"

saver.export_meta_graph(ckpt_json_path, as_text=True)
```

输出：

[https://daiwk.github.io/assets/tf.saver.demo.meta.json](ttps://daiwk.github.io/assets/tf.saver.demo.meta.json)

### meta_info_def

如上，MetaInfoDef中包含如下信息：

```c++
  message MetaInfoDef {
    // User specified Version string. Can be the name of the model and revision,
    // steps this model has been trained to, etc.
    string meta_graph_version = 1;

    // A copy of the OpDefs used by the producer of this graph_def.
    // Descriptions and Ops not used in graph_def are stripped out.
    OpList stripped_op_list = 2;

    // A serialized protobuf. Can be the time this meta graph is created, or
    // modified, or name of the model.
    google.protobuf.Any any_info = 3;

    // User supplied tag(s) on the meta_graph and included graph_def.
    //
    // MetaGraphDefs should be tagged with their capabilities or use-cases.
    // Examples: "train", "serve", "gpu", "tpu", etc.
    // These tags enable loaders to access the MetaGraph(s) appropriate for a
    // specific use-case or runtime environment.
    repeated string tags = 4;

    // The __version__ string of the tensorflow build used to write this graph.
    // This will be populated by the framework, which will overwrite any user
    // supplied value.
    string tensorflow_version = 5;

    // The __git_version__ string of the tensorflow build used to write this
    // graph. This will be populated by the framework, which will overwrite any
    // user supplied value.
    string tensorflow_git_version = 6;

    // A flag to denote whether default-valued attrs have been stripped from
    // the nodes in this graph_def.
    bool stripped_default_attrs = 7;
  }
```

### graph_def

### saver_def

### collection_def

