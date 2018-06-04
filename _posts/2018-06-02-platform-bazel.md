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


