---
layout: post
category: "platform"
title: "bazel"
tags: [bazel, ]
---

目录

<!-- TOC -->

- [c++](#c)

<!-- /TOC -->


## c++

下载demo

```shell
git clone https://github.com/bazelbuild/examples/
```

```shell
examples
└── cpp-tutorial
    ├──stage1
    │  ├── main
    │  │   ├── BUILD
    │  │   └── hello-world.cc
    │  └── WORKSPACE
    ├──stage2
    │  ├── main
    │  │   ├── BUILD
    │  │   ├── hello-world.cc
    │  │   ├── hello-greet.cc
    │  │   └── hello-greet.h
    │  └── WORKSPACE
    └──stage3
       ├── main
       │   ├── BUILD
       │   ├── hello-world.cc
       │   ├── hello-greet.cc
       │   └── hello-greet.h
       ├── lib
       │   ├── BUILD
       │   ├── hello-time.cc
       │   └── hello-time.h
       └── WORKSPACE
```

+ 在project的根目录有一个WORKSPACE文件
+ 有一个BUILD文件的目录是一个package

常用命令

+ build WORKSPACE 下面所有的 target，会扫所有的目录

```shell
bazel build //...
```


+ 单独的 build 的 target 则直接 //:demo， 这里 :demo 是 target name

```shell
bazel build //:demo
```

+ 执行 :demo

```shell
bazel run //:demo
```

+ 跑 demo_test 测试

```shell
baze test //:demo_test
```

