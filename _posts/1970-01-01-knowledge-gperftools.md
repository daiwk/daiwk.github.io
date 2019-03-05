---
layout: post
category: "knowledge"
title: "cpu、内存问题排查（gperftools、perf等）"
tags: [gperftools, perf, 火焰图]
---

目录

<!-- TOC -->

- [内存泄漏](#%E5%86%85%E5%AD%98%E6%B3%84%E6%BC%8F)
- [常用工具](#%E5%B8%B8%E7%94%A8%E5%B7%A5%E5%85%B7)
  - [gperftools的功能支持](#gperftools%E7%9A%84%E5%8A%9F%E8%83%BD%E6%94%AF%E6%8C%81)
  - [perf+火焰图](#perf%E7%81%AB%E7%84%B0%E5%9B%BE)
    - [火焰图工具](#%E7%81%AB%E7%84%B0%E5%9B%BE%E5%B7%A5%E5%85%B7)
    - [perf](#perf)

<!-- /TOC -->

参考[高阶干货\｜如何用gperftools分析深度学习框架的内存泄漏问题](https://mp.weixin.qq.com/s?__biz=MzIxNTgyMDMwMw==&mid=2247484403&idx=1&sn=5b260e7d681a4550811ee5611a1dd4ce&chksm=97933293a0e4bb85104066213606f5c89002fe78fa0518d95c71918a5a6ae656c62beac0f1ae&mpshare=1&scene=1&srcid=0608zWg71wg9411XBaYQ7o1V&pass_ticket=xLsJxSJh9Kgj4HKrq0S6VH1cKTCnSBShWGuwGJy9Gfpbp1CgoA6crqJiPhq9JjnM#rd)

## 内存泄漏

内存泄漏一般是由于程序**在堆(heap)上**分配了内存而没有释放，随着程序的运行占用的内存越来越大，一方面会影响程序的稳定性，可能让运行速度越来越慢，或者造成oom，甚至会影响程序所运行的机器的稳定性，造成宕机。

## 常用工具

+ valgrind直接分析非常困难，需要自己编译debug版本的、带valgrind支持的专用Python版本，而且输出的信息中大部分是Python自己的符号和调用信息，很难看出有用的信息，另外**使用valgrind会让程序运行速度变得非常慢**，所以不建议使用。
+ gperftools使用简单，无需重新编译代码即可运行，对运行速度的影响也比较小。

### gperftools的功能支持

gperftool主要支持以下四个功能：

+ thread-caching malloc
+ heap-checking using tcmalloc
+ heap-profiling using tcmalloc
+ CPU profiler

### perf+火焰图

#### 火焰图工具

[https://github.com/brendangregg/FlameGraph](https://github.com/brendangregg/FlameGraph)下载下来

然后解压并改个名：

```shell
unzip master.zip
mv ./FlameGraph-master/ ./FlameGraph
```

#### perf

如果内核版本2.6.32_1-23-0-0以上，系统有自带的(centos6u3以上的也自带)。否则需要安装：

```shell
wget https://cdn.kernel.org/pub/linux/kernel/v4.x/linux-4.12.9.tar.xz
xz -d linux-4.12.9.tar.xz
tar xf linux-4.12.9.tar
cd linux-4.12.9/tools/perf
export PATH=/opt/compiler/gcc-4.8.2/bin:/opt/stap/bin:$PATH
make && sudo make install
```

```shell
perf record -F 500 -p $task_id -o perf.data -g sleep $time &
wait
## 参数
##record - Run a command and record its profile into perf.data
##-F，--freq= ，Profile at this frequency.
##-p， --pid=，Record events on existing process ID
##-o，--output=，Output file name.
##-g，--call-graph，Do call-graph (stack chain/backtrace) recording.
##sleep，采集时长，单位s
```

然后

```shell
perf script -i perf.data > out.perf
./FlameGraph/stackcollapse-perf.pl out.perf > out.folded
./FlameGraph/flamegraph.pl out.folded > cpu.svg
```

例如，我们对某进程搞一下，可以看到以下火焰图，然后我们发现xgboost的预测```xgboost::predictor::CPUPredictor::PredLoopSpecalize```就占了总cpu利用的9.36%！。。：

<html>
<br/>
<img src='../assets/flame-perf.png' style='max-width: 250px'/>
<br/>
</html>
