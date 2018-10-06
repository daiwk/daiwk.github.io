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

其中的meta_graph_version(计算图的版本号)、tags(用户指定的一些标签)，如果没在saver中指定，都默认是空。stripped_op_list属性记录了计算图上使用的所有**运算方法**的信息。如果某个运算在计算图中出现多次，则其在stripped_op_list中也**只会出现一次**。stripped_op_list的类型是OpList其定义在```tensorflow/core/framework/op_def.proto```中，如下：

```c++
syntax = "proto3";

package tensorflow;
option cc_enable_arenas = true;
option java_outer_classname = "OpDefProtos";
option java_multiple_files = true;
option java_package = "org.tensorflow.framework";
option go_package = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework";
import "tensorflow/core/framework/attr_value.proto";
import "tensorflow/core/framework/types.proto";

// Defines an operation. A NodeDef in a GraphDef specifies an Op by
// using the "op" field which should match the name of a OpDef.
// LINT.IfChange
message OpDef {
  // Op names starting with an underscore are reserved for internal use.
  // Names should be CamelCase and match the regexp "[A-Z][a-zA-Z0-9_]*".
  string name = 1;

  // For describing inputs and outputs.
  message ArgDef {
    // Name for the input/output.  Should match the regexp "[a-z][a-z0-9_]*".
    string name = 1;

    // Human readable description.
    string description = 2;

    // Describes the type of one or more tensors that are accepted/produced
    // by this input/output arg.  The only legal combinations are:
    // * For a single tensor: either the "type" field is set or the
    //   "type_attr" field is set to the name of an attr with type "type".
    // * For a sequence of tensors with the same type: the "number_attr"
    //   field will be set to the name of an attr with type "int", and
    //   either the "type" or "type_attr" field will be set as for
    //   single tensors.
    // * For a sequence of tensors, the "type_list_attr" field will be set
    //   to the name of an attr with type "list(type)".
    DataType type = 3;
    string type_attr = 4;    // if specified, attr must have type "type"
    string number_attr = 5;  // if specified, attr must have type "int"
    // If specified, attr must have type "list(type)", and none of
    // type, type_attr, and number_attr may be specified.
    string type_list_attr = 6;

    // For inputs: if true, the inputs are required to be refs.
    //   By default, inputs can be either refs or non-refs.
    // For outputs: if true, outputs are refs, otherwise they are not.
    bool is_ref = 16;
  };

  // Description of the input(s).
  repeated ArgDef input_arg = 2;

  // Description of the output(s).
  repeated ArgDef output_arg = 3;

  // Description of the graph-construction-time configuration of this
  // Op.  That is to say, this describes the attr fields that will
  // be specified in the NodeDef.
  message AttrDef {
    // A descriptive name for the argument.  May be used, e.g. by the
    // Python client, as a keyword argument name, and so should match
    // the regexp "[a-z][a-z0-9_]+".
    string name = 1;

    // One of the type names from attr_value.proto ("string", "list(string)",
    // "int", etc.).
    string type = 2;

    // A reasonable default for this attribute if the user does not supply
    // a value.  If not specified, the user must supply a value.
    AttrValue default_value = 3;

    // Human-readable description.
    string description = 4;

    // TODO(josh11b): bool is_optional?

    // --- Constraints ---
    // These constraints are only in effect if specified.  Default is no
    // constraints.

    // For type == "int", this is a minimum value.  For "list(___)"
    // types, this is the minimum length.
    bool has_minimum = 5;
    int64 minimum = 6;

    // The set of allowed values.  Has type that is the "list" version
    // of the "type" field above (uses the "list" field of AttrValue).
    // If type == "type" or "list(type)" above, then the "type" field
    // of "allowed_values.list" has the set of allowed DataTypes.
    // If type == "string" or "list(string)", then the "s" field of
    // "allowed_values.list" has the set of allowed strings.
    AttrValue allowed_values = 7;
  }
  repeated AttrDef attr = 4;

  // Optional deprecation based on GraphDef versions.
  OpDeprecation deprecation = 8;

  // One-line human-readable description of what the Op does.
  string summary = 5;

  // Additional, longer human-readable description of what the Op does.
  string description = 6;

  // -------------------------------------------------------------------------
  // Which optimizations this operation can participate in.

  // True if the operation is commutative ("op(a,b) == op(b,a)" for all inputs)
  bool is_commutative = 18;

  // If is_aggregate is true, then this operation accepts N >= 2
  // inputs and produces 1 output all of the same type.  Should be
  // associative and commutative, and produce output with the same
  // shape as the input.  The optimizer may replace an aggregate op
  // taking input from multiple devices with a tree of aggregate ops
  // that aggregate locally within each device (and possibly within
  // groups of nearby devices) before communicating.
  // TODO(josh11b): Implement that optimization.
  bool is_aggregate = 16;  // for things like add

  // Other optimizations go here, like
  //   can_alias_input, rewrite_when_output_unused, partitioning_strategy, etc.

  // -------------------------------------------------------------------------
  // Optimization constraints.

  // Ops are marked as stateful if their behavior depends on some state beyond
  // their input tensors (e.g. variable reading op) or if they have
  // a side-effect (e.g. printing or asserting ops). Equivalently, stateless ops
  // must always produce the same output for the same input and have
  // no side-effects.
  //
  // By default Ops may be moved between devices.  Stateful ops should
  // either not be moved, or should only be moved if that state can also
  // be moved (e.g. via some sort of save / restore).
  // Stateful ops are guaranteed to never be optimized away by Common
  // Subexpression Elimination (CSE).
  bool is_stateful = 17;  // for things like variables, queue

  // -------------------------------------------------------------------------
  // Non-standard options.

  // By default, all inputs to an Op must be initialized Tensors.  Ops
  // that may initialize tensors for the first time should set this
  // field to true, to allow the Op to take an uninitialized Tensor as
  // input.
  bool allows_uninitialized_input = 19;  // for Assign, etc.
};
// LINT.ThenChange(
//     https://www.tensorflow.org/code/tensorflow/core/framework/op_def_util.cc)

// Information about version-dependent deprecation of an op
message OpDeprecation {
  // First GraphDef version at which the op is disallowed.
  int32 version = 1;

  // Explanation of why it was deprecated and what to use instead.
  string explanation = 2;
};

// A collection of OpDefs
message OpList {
  repeated OpDef op = 1;
};

```

例如，如下就是名为Add的运算。两个input_arg，一个output_arg，它们都有type_attr，且值均为T。所以在attr中，必须出现name是T的属性，及其allow_values。

```shell
    op {
      name: "Add"
      input_arg {
        name: "x"
        type_attr: "T"
      }
      input_arg {
        name: "y"
        type_attr: "T"
      }
      output_arg {
        name: "z"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_BFLOAT16
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_UINT8
            type: DT_INT8
            type: DT_INT16
            type: DT_INT32
            type: DT_INT64
            type: DT_COMPLEX64
            type: DT_COMPLEX128
            type: DT_STRING
          }
        }
      }
    }
```

另外，在meta_info_def中，还有如下记录生成当前计算图的tensorflow版本的属性：

```c++
  tensorflow_version: "1.9.0"
  tensorflow_git_version: "v1.9.0-0-g25c197e023"
```

### graph_def

在```tensorflow/core/framework/graph.proto```中定义了GraphDef，用于记录计算图上的节点信息。每个节点对应一个运算。在meta_info_def中已包含了所有运算的信息，所以graph_def**只关注运算的连接结构**。GraphDef中的versions比较简单，主要存储tf的版本号，主要信息都在NodeDef类型的node中。

```c++
syntax = "proto3";

package tensorflow;
option cc_enable_arenas = true;
option java_outer_classname = "GraphProtos";
option java_multiple_files = true;
option java_package = "org.tensorflow.framework";
option go_package = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework";
import "tensorflow/core/framework/node_def.proto";
import "tensorflow/core/framework/function.proto";
import "tensorflow/core/framework/versions.proto";

// Represents the graph of operations
message GraphDef {
  repeated NodeDef node = 1;

  // Compatibility versions of the graph.  See core/public/version.h for version
  // history.  The GraphDef version is distinct from the TensorFlow version, and
  // each release of TensorFlow will support a range of GraphDef versions.
  VersionDef versions = 4;

  // Deprecated single version field; use versions above instead.  Since all
  // GraphDef changes before "versions" was introduced were forward
  // compatible, this field is entirely ignored.
  int32 version = 3 [deprecated = true];

  // EXPERIMENTAL. DO NOT USE OR DEPEND ON THIS YET.
  //
  // "library" provides user-defined functions.
  //
  // Naming:
  //   * library.function.name are in a flat namespace.
  //     NOTE: We may need to change it to be hierarchical to support
  //     different orgs. E.g.,
  //     { "/google/nn", { ... }},
  //     { "/google/vision", { ... }}
  //     { "/org_foo/module_bar", { ... }}
  //     map<string, FunctionDefLib> named_lib;
  //   * If node[i].op is the name of one function in "library",
  //     node[i] is deemed as a function call. Otherwise, node[i].op
  //     must be a primitive operation supported by the runtime.
  //
  //
  // Function call semantics:
  //
  //   * The callee may start execution as soon as some of its inputs
  //     are ready. The caller may want to use Tuple() mechanism to
  //     ensure all inputs are ready in the same time.
  //
  //   * The consumer of return values may start executing as soon as
  //     the return values the consumer depends on are ready.  The
  //     consumer may want to use Tuple() mechanism to ensure the
  //     consumer does not start until all return values of the callee
  //     function are ready.
  FunctionDefLibrary library = 2;
};

```

其中的NodeDef在```tensorflow/core/framework/node_def.proto```中定义如下：

+ name是节点名称，是一个节点的唯一标识符。tf中可以通过节点名称来获取相应的节点。
+ op属性给出了该节点使用的tf运算方法的名称，通过此名称可以在计算图元图的meta_info_def中找到该运算的具体信息。
+ input属性中每个字符串的取值格式为node:src_output
  + node部分给出一个节点的名称
  + src_output部分表明这个输入是指定节点的**第几个输出**。src_output为0时可以省略，即node:0可以记为node。
+ device属性

```c++
syntax = "proto3";

package tensorflow;
option cc_enable_arenas = true;
option java_outer_classname = "NodeProto";
option java_multiple_files = true;
option java_package = "org.tensorflow.framework";
option go_package = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework";
import "tensorflow/core/framework/attr_value.proto";

message NodeDef {
  // The name given to this operator. Used for naming inputs,
  // logging, visualization, etc.  Unique within a single GraphDef.
  // Must match the regexp "[A-Za-z0-9.][A-Za-z0-9_./]*".
  string name = 1;

  // The operation name.  There may be custom parameters in attrs.
  // Op names starting with an underscore are reserved for internal use.
  string op = 2;

  // Each input is "node:src_output" with "node" being a string name and
  // "src_output" indicating which output tensor to use from "node". If
  // "src_output" is 0 the ":0" suffix can be omitted.  Regular inputs
  // may optionally be followed by control inputs that have the format
  // "^node".
  repeated string input = 3;

  // A (possibly partial) specification for the device on which this
  // node should be placed.
  // The expected syntax for this string is as follows:
  //
  // DEVICE_SPEC ::= PARTIAL_SPEC
  //
  // PARTIAL_SPEC ::= ("/" CONSTRAINT) *
  // CONSTRAINT ::= ("job:" JOB_NAME)
  //              | ("replica:" [1-9][0-9]*)
  //              | ("task:" [1-9][0-9]*)
  //              | ("device:" [A-Za-z]* ":" ([1-9][0-9]* | "*") )
  //
  // Valid values for this string include:
  // * "/job:worker/replica:0/task:1/device:GPU:3"  (full specification)
  // * "/job:worker/device:GPU:3"                   (partial specification)
  // * ""                                    (no specification)
  //
  // If the constraints do not resolve to a single device (or if this
  // field is empty or not present), the runtime will attempt to
  // choose a device automatically.
  string device = 4;

  // Operation-specific graph-construction-time configuration.
  // Note that this should include all attrs defined in the
  // corresponding OpDef, including those with a value matching
  // the default -- this allows the default to change and makes
  // NodeDefs easier to interpret on their own.  However, if
  // an attr with a default is not specified in this list, the
  // default will be used.
  // The "names" (keys) must match the regexp "[a-z][a-z0-9_]+" (and
  // one of the names from the corresponding OpDef's attr field).
  // The values must have a type matching the corresponding OpDef
  // attr's type field.
  // TODO(josh11b): Add some examples here showing best practices.
  map<string, AttrValue> attr = 5;
};

```



### saver_def

### collection_def

