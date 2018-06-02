---
layout: post
category: "platform"
title: "grpc"
tags: [grpc, ]
---

目录

<!-- TOC -->

- [安装](#)
- [c++ tutorial](#c-tutorial)
    - [快速c++入门](#c)
    - [进阶](#)

<!-- /TOC -->

[https://github.com/grpc/grpc](https://github.com/grpc/grpc)

## 安装

[https://github.com/grpc/grpc/blob/master/INSTALL.md](https://github.com/grpc/grpc/blob/master/INSTALL.md)

```shell
git clone -b $(curl -L https://grpc.io/release) https://github.com/grpc/grpc
cd grpc
git submodule update --init
make
[sudo] make install
```

然后



## c++ tutorial

### 快速c++入门

[https://github.com/grpc/grpc/tree/master/examples/cpp](https://github.com/grpc/grpc/tree/master/examples/cpp)

先安装protoc

```shell
cd third_party/protobuf/
make
make install
```

然后编译demo

```shell
cd examples/cpp/helloworld
make
```

启动server

```shell
./greeter_server
./greeter_async_server # 异步
```

启动client

```shell
greeter_client
greeter_async_client #异步
greeter_async_client2 #异步
```

### 进阶

[https://github.com/grpc/grpc/blob/master/examples/cpp/cpptutorial.md](https://github.com/grpc/grpc/blob/master/examples/cpp/cpptutorial.md)

