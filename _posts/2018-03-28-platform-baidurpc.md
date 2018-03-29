---
layout: post
category: "platform"
title: "baidurpc"
tags: [baidurpc, ]
---

目录

<!-- TOC -->

- [简介](#简介)
    - [baidurpc的动机](#baidurpc的动机)
    - [key components](#key-components)
        - [rpc::socket](#rpcsocket)
            - [使fd ABA-free，且能**原子**地**写入消息**](#使fd-aba-free且能原子地写入消息)

<!-- /TOC -->

[https://github.com/brpc/brpc](https://github.com/brpc/brpc)

## 简介

### baidurpc的动机

现有rpc不好用的原因
+ 不透明：没开源，所有有一点相关的问题都想找rpc团队跟进
+ 难扩展：很难被使用超过2年
+ 性能差：在机器较忙或者混部环境下波动很大

baidurpc的解决方法
+ /vars, /flags, /rpcz等http内置服务，方便快速排查问题
+ 重视接口设计。支持了百度所有rpc协议和多种外部协议，共12种
+ 充分考虑多线程，尽量规避全局竞争。较其他rpc大大提升了性能。

### key components

#### rpc::socket

##### 使fd ABA-free，且能**原子**地**写入消息**

+ 使fd ABA-free：
linux管理fd时用的是32位的整数，但关掉之后就释放了，可能被另一个进程打开，而指向了另一个设备。所以多线程时，可能同一个整数被多个线程使用，用的时候可能fd已经被关掉或者被重复打开（ABA-一开始是a，后来变b，可能又变回a）
+ 能原子地写入消息：
xxx