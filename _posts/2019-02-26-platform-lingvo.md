---
layout: post
category: "platform"
title: "lingvo"
tags: [lingvo, ]
---

目录

<!-- TOC -->

- [安装&使用](#%E5%AE%89%E8%A3%85%E4%BD%BF%E7%94%A8)

<!-- /TOC -->


代码：[https://github.com/tensorflow/lingvo](https://github.com/tensorflow/lingvo)

论文：[Lingvo: a Modular and Scalable Framework for Sequence-to-Sequence Modeling](https://arxiv.org/abs/1902.08295)

Lingvo 是一个能够为协作式深度学习研究提供完整解决方案的 Tensorflow 框架，尤其关注序列到序列模型。Lingvo 模型由模块化构件组成，这些构件灵活且易于扩展，实验配置集中且可定制。分布式训练和量化推理直接在框架内得到支持，框架内包含大量 utilities、辅助函数和最新研究思想的现有实现。

设计原则如下：

+ 单个代码块应该精细且**模块化**，它们会使用相同的接口，同时也**容易扩展**；
+ 实验应该是共享的、可比较的、可复现的、可理解的和正确的；
+ 性能应该可以**高效地扩展到生产规模的数据集**，或拥有**数百个加速器的分布式训练系统**；
+ 当模型从**研究转向产品**时应该尽可能共享代码。

## 安装&使用

首先的首先，需要安装：

```shell
pip install tf-nightly
pip install tensorflow -U
```

首先下载数据集(如果遇到下载的ssl问题，可以参考[https://daiwk.github.io/posts/knowledge-tf-usage.html#%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98](https://daiwk.github.io/posts/knowledge-tf-usage.html#%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98))

```shell
mkdir -p /tmp/mnist
bazel run -c opt //lingvo/tools:keras2ckpt -- --dataset=mnist --out=/tmp/mnist/mnist
```

然后build一个trainer，如果出现如下错误。。把装了tf的py扔到PATH里就行。

```shell
bazel build -c opt //lingvo:trainer
ERROR: /home/disk2/daiwenkai/workspaces/tf/lingvo/lingvo/core/ops/BUILD:24:1: no such package '@tensorflow_solib//': Traceback (most recent call last):
        File "/home/disk2/daiwenkai/workspaces/tf/lingvo/lingvo/repo.bzl", line 88
                _find_tf_lib_path(repo_ctx)
        File "/home/disk2/daiwenkai/workspaces/tf/lingvo/lingvo/repo.bzl", line 30, in _find_tf_lib_path
                fail("Could not locate tensorflow ins...")
Could not locate tensorflow installation path. and referenced by '//lingvo/core/ops:x_ops'
ERROR: Analysis of target '//lingvo:trainer' failed; build aborted: no such package '@tensorflow_solib//': Traceback (most recent call last):
        File "/home/disk2/daiwenkai/workspaces/tf/lingvo/lingvo/repo.bzl", line 88
                _find_tf_lib_path(repo_ctx)
        File "/home/disk2/daiwenkai/workspaces/tf/lingvo/lingvo/repo.bzl", line 30, in _find_tf_lib_path
                fail("Could not locate tensorflow ins...")
Could not locate tensorflow installation path.
INFO: Elapsed time: 5.916s
INFO: 0 processes.
```

想看详细日志，可以：

```shell
bazel build -c opt //lingvo:trainer --sandbox_debug
```

想把当前的PATH加进去，可以：

```shell
bazel build -c opt //lingvo:trainer --sandbox_debug --action_env=PATH
```

这样，就会显示：

```shell
ERROR: /home/disk2/daiwenkai/workspaces/tf/lingvo/lingvo/core/ops/BUILD:304:1: Executing genrule //lingvo/core/ops:hyps_proto_gencc failed (Exit 1) process-wrapper failed: error executing command 
  (cd /home/work/.cache/bazel/_bazel_work/6019387a2835fd5f247d5cbc29b5ee5a/execroot/__main__ && \
  exec env - \
    PATH=/opt/compiler/gcc-4.8.2/bin/:/home/disk2/daiwenkai/tools/python-2.7.14/bin/:/home/disk2/daiwenkai/workspaces/tf/prepare/:/home/disk2/daiwenkai/workspaces/tf/prepare/jdk1.8.0_152/bin:/home/work/.hmpclient/bin:/home/work/.BCloud/bin:/home/work/.hmpclient/bin:/home/work/.jumbo/opt/sun-java6/bin:/home/work/.jumbo/opt/sun-java6/jre/bin:/home/work/.jumbo/opt/sun-java6/db/bin:/home/work/.jumbo/bin/core_perl:/home/work/.jumbo/bin:/usr/kerberos/bin:/usr/local/bin:/bin:/usr/bin:/usr/X11R6/bin:/opt/bin:/home/opt/bin \
    TMPDIR=/tmp \
  /home/work/.cache/bazel/_bazel_work/install/8e122cdaf21df7dee88c59e8c0fa6061/_embedded_binaries/process-wrapper '--timeout=0' '--kill_delay=15' /bin/bash -c 'source external/bazel_tools/tools/genrule/genrule-setup.sh; 
          mkdir -p bazel-out/k8-opt/genfiles/lingvo/core/ops/tf_proto.$$;
          tar -C bazel-out/k8-opt/genfiles/lingvo/core/ops/tf_proto.$$ -xf bazel-out/host/genfiles/lingvo/tf_protos.tar;
          external/protobuf_protoc/bin/protoc --proto_path=bazel-out/k8-opt/genfiles/lingvo/core/ops/tf_proto.$$  --proto_path=. --cpp_out=bazel-out/k8-opt/genfiles lingvo/core/ops/hyps.proto;
          rm -rf bazel-out/k8-opt/genfiles/lingvo/core/ops/tf_proto.$$
        ')
external/protobuf_protoc/bin/protoc: /lib64/tls/libc.so.6: version `GLIBC_2.4' not found (required by external/protobuf_protoc/bin/protoc)
Target //lingvo:trainer failed to build
Use --verbose_failures to see the command lines of failed build steps.
```

这个时候我们会发现，，因为依赖的是用gcc4.8编译好的protoc，而下下来本地跑bazel的时候，会用本地默认的lib64，所以不行，所以我们用trick来搞，对```lingvo/lingvo.bzl```进行如下修改，也就是

+ 一方面把```/opt/compiler/gcc-4.8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib```加到protoc前面，
+ 另一方面，手动把protc的zip下载下来，例如```lingvo/repos.bzl```里要求的是3.6.1版本，然后解压到tmp目录，所以咱们把里面的include目录(里面有一堆.proto文件)加进来『```-I/home/disk2/daiwenkai/workspaces/tf/lingvo/tmp/include```』

```python
def _proto_gen_cc_src(name, basename):
    native.genrule(
        name = name,
        srcs = [basename + ".proto"],
        outs = [basename + ".pb.cc", basename + ".pb.h"],
        tools = [
            "@protobuf_protoc//:protoc_bin",
            "//lingvo:tf_dot_protos",
        ],      
        # TODO(drpng): only unpack if tf_proto dependency is requested.
        cmd = """
          mkdir -p $(@D)/tf_proto.$$$$;
          tar -C $(@D)/tf_proto.$$$$ -xf $(location //lingvo:tf_dot_protos);
          /opt/compiler/gcc-4.8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib $(location @protobuf_protoc//:protoc_bin) --proto_path=$(@D)/tf_proto.$$$$ -I/home/disk2/daiwenkai/workspaces/tf/lingvo/tmp/include --proto
_path=. --cpp_out=$(GENDIR) $(<);
          rm -rf $(@D)/tf_proto.$$$$
        """,
    )

def _proto_gen_py_src(name, basename):
    native.genrule(
        name = name, 
        srcs = [basename + ".proto"],
        outs = [basename + "_pb2.py"],
        tools = [
            "@protobuf_protoc//:protoc_bin",
            "//lingvo:tf_dot_protos",
        ],      
        # TODO(drpng): only unpack if tf_proto dependency is requested.
        cmd = """
          mkdir -p $(@D)/tf_proto.$$$$;
          tar -C $(@D)/tf_proto.$$$$ -xf $(location //lingvo:tf_dot_protos);
          /opt/compiler/gcc-4.8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib $(location @protobuf_protoc//:protoc_bin) --proto_path=$(@D)/tf_proto.$$$$ -I/home/disk2/daiwenkai/workspaces/tf/lingvo/tmp/include --proto
_path=. --python_out=$(GENDIR) $(<);
          rm -rf $(@D)/tf_proto.$$$$
        """,
    )
```

注意！！！这里的protobuf版本要和你装的tf的protobuf版本一样。。。比如你的tf是3.6.0的，那就要把这里依赖的改成3.6.0的protoc~~3.6.1是不行的呢！！

然后呢。。。发现我们从tf源码build出来的include里会出现这种问题[https://github.com/tensorflow/lingvo/issues/39](https://github.com/tensorflow/lingvo/issues/39)，于是。。
