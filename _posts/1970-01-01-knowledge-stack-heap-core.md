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

已经core了的文件好像查不了mappings，用gdb启动的可以看：

```shell
info proc  mappings
```

前面会有：

```shell
(gdb) info proc  mappings   
process 544  
Mapped address spaces:  
  
    Start Addr   End Addr       Size     Offset objfile  
        0x8000     0x9000     0x1000        0x0 /mnt/test_class  
       0x10000    0x11000     0x1000        0x0 /mnt/test_class  
       0x11000    0x32000    0x21000        0x0 [heap]  
    0xb6d39000 0xb6e64000   0x12b000        0x0 /lib/libc-2.19.so  
    0xb6e64000 0xb6e6c000     0x8000   0x12b000 /lib/libc-2.19.so  
```

说明0x11000-0x32000这总共0x21000的大小是堆空间

最后面会有：

```shell
      0x7ffff7ffb000     0x7ffff7ffc000     0x1000        0x0 [vdso]
      0x7ffff7ffc000     0x7ffff7ffd000     0x1000    0x20000 /home/opt/gcc-4.8.2.bpkg-r4/gcc-4.8.2.bpkg-r4/lib64/ld-2.18.so
      0x7ffff7ffd000     0x7ffff7ffe000     0x1000    0x21000 /home/opt/gcc-4.8.2.bpkg-r4/gcc-4.8.2.bpkg-r4/lib64/ld-2.18.so
      0x7ffff7ffe000     0x7ffff7fff000     0x1000        0x0 
      0x7ffffff73000     0x7ffffffff000    0x8c000        0x0 [stack]
  0xffffffffff600000 0xffffffffff601000     0x1000        0x0 [vsyscall]
```

说明0x7ffffff73000-0x7ffffffff000这总共0x8c000=789999=789k=0.8MB的大小是栈空间？？好像不太对呢。。。

查看当前frame：

```shell
info frame
Stack level 1, frame at 0x7f7ed3284310:
 rip = 0x7f7ed6da1f50 in nerl::NerlPlus::tagging (baidu/xxxx/src/dddd.cpp:599); saved rip = 0x7f7ed6d9964e
 called by frame at 0x7f7ed3284360, caller of frame at 0x7f7ed32841e0
 source language c++.
 Arglist at 0x7f7ed32841d8, args: this=0x2fd6950, iTokens=0x7f7caae7f010, iTokensCount=1, iNerlBuff=0x7f7c9f706710, tmpTags=..., oTags=..., 
    flags=nerl::DEFAULT_FLAGS
 Locals at 0x7f7ed32841d8, Previous frame's sp is 0x7f7ed3284310
 Saved registers:
  rbx at 0x7f7ed32842d8, rbp at 0x7f7ed32842e0, r12 at 0x7f7ed32842e8, r13 at 0x7f7ed32842f0, r14 at 0x7f7ed32842f8, r15 at 0x7f7ed3284300,
  rip at 0x7f7ed3284308
```

## 堆、栈导致的core

### 栈空间不足

参考：[https://blog.csdn.net/u011866460/article/details/42525171](https://blog.csdn.net/u011866460/article/details/42525171)

例如，程序中有两个大小为`\(2048*2048\)`的char数组，算下来，一个char是一个字节，两个`\(2048*2048\)`的数组便是`\(2*2048*2048=8388608=8*1024*1024=8MB\)`的空间。所以，如果这个时候还有别的栈上的变量，而栈空间如果 只有8MB，那么，就会core!!!

linux限制了栈空间大小，自己定义的变量都是在栈空间上分配的，子函数在调用时才会装入栈中，当定义的变量过大则会超出栈空间，从而段错误。所以，尽可能使用堆空间，比如用new malloc vector等
