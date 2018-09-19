---
layout: post
category: "platform"
title: "leveldb rocksdb anna等"
tags: [leveldb, rocksdb, anna, ]
---

目录

<!-- TOC -->

- [leveldb](#leveldb)
- [rocksdb](#rocksdb)
- [ssd简介](#ssd%E7%AE%80%E4%BB%8B)
- [Cassandra](#cassandra)
- [anna](#anna)

<!-- /TOC -->


## leveldb

memtable + wal文件 +l0-7


## rocksdb



## ssd简介

flash:写page 擦除block，512k一个block。擦除目前最好的是3w次

ssd中大多数厂商是通过一个映射表，把物理映射到逻辑上，将随机读改成顺序读

每个ssd会留出一块block做缓存，用户不可见。当用户区写满时，会把有用的数据移到这块block，然后移回去。如果失效数据很少，写放大很严重（目前写放大是10倍左右）


ssd写性能: 顺序500m/s，随机一般要/10
ssd读性能：约3G/s


xxxx存储系统针对ssd设计：有个全内存的index，是个hash，然后也有个磁盘的wal。内存里只记录偏移。设计上是顺序写

id是64int，其中前几位是标记offset，导致只有约16个bit可用==》最多只能用128g磁盘。但现在磁盘是800-1024t，所以要充分利用得部署多个引擎。但这样就变成了随机写


结合：
单机引擎部分：用leveldb只做index，存储key+offset，所以不受限于内存，解决xxx的问题。

分布式：主从复制，写只写master，从只读


新硬件AEP：往内存口插

## Cassandra



## anna

[秒杀Redis的KVS上云了！伯克利重磅开源Anna 1.0](https://mp.weixin.qq.com/s?__biz=MjM5MDE0Mjc4MA==&mid=2651009148&idx=2&sn=a42a12c16c3e08bfe8a06b0ac347663f&chksm=bdbec82f8ac941390b86eea1a438383c823054b173857e3299794d2098b32bbefb6a5562cde8&scene=27#wechat_redirect)

berkely anna kv: 针对分布式的改造。

正常主-从group一般500个，但一开始hash要搞3w多个，方便扩容

节点没有主从概念，但每个slot有主从
一致性：通过client端做一致性，很容易支持多种一致性（最终一致、强一致等）


