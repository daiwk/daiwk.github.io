---
layout: post
category: "knowledge"
title: "栈、堆、core等"
tags: [stack, heap, core, 栈, 堆, ]
---

目录

<!-- TOC -->

- [基础命令](#基础命令)
- [堆与栈](#堆与栈)
- [core](#core)
- [堆、栈导致的core](#堆栈导致的core)
    - [栈空间不足](#栈空间不足)

<!-- /TOC -->

## 基础命令

```shell
ulimit -a
core file size          (blocks, -c) unlimited
data seg size           (kbytes, -d) unlimited
file size               (blocks, -f) unlimited
pending signals                 (-i) 1031511
max locked memory       (kbytes, -l) 64
max memory size         (kbytes, -m) unlimited
open files                      (-n) 10240
pipe size            (512 bytes, -p) 8
POSIX message queues     (bytes, -q) 819200
stack size              (kbytes, -s) 10240
cpu time               (seconds, -t) unlimited
max user processes              (-u) 1031511
virtual memory          (kbytes, -v) unlimited
file locks                      (-x) unlimited
```

可见，stack size即栈空间的大小是10240KB，也就是10MB。可用ulimit -s可以只看栈空间大小。

## 堆与栈

## core

[https://blog.csdn.net/caspiansea/article/details/24450377](https://blog.csdn.net/caspiansea/article/details/24450377)

```
info proc  mappings
```

## 堆、栈导致的core

### 栈空间不足

参考：[https://blog.csdn.net/u011866460/article/details/42525171](https://blog.csdn.net/u011866460/article/details/42525171)

例如，程序中有两个大小为`\(2048*2048\)`的char数组，算下来，一个char是一个字节，两个`\(2048*2048\)`的数组便是`\(2*2048*2048=8388608=8*1024*1024=8MB\)`的空间。所以，如果这个时候还有别的栈上的变量，而栈空间如果 只有8MB，那么，就会core!!!

linux限制了栈空间大小，自己定义的变量都是在栈空间上分配的，子函数在调用时才会装入栈中，当定义的变量过大则会超出栈空间，从而段错误。所以，尽可能使用堆空间，比如用new malloc vector等
