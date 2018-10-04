---
layout: post
category: "platform"
title: "kubernetes-container"
tags: [kubernetes, k8s, container ]
---

目录

<!-- TOC -->

- [Namespace](#namespace)
- [Cgroups](#cgroups)
- [rootfs文件系统](#rootfs%E6%96%87%E4%BB%B6%E7%B3%BB%E7%BB%9F)

<!-- /TOC -->

容器技术的核心功能，就是通过**约束和修改进程**的动态表现，从而为其创造一个『边界』。

+ Namespace技术用来修改进程视图
+ Cgroups技术用来制造约束

## Namespace

参考[05 \| 白话容器基础（一）：从进程说开去](https://time.geekbang.org/column/article/14642)

**进程**：一个程序运行后的计算机执行环境（磁盘上的可执行文件、内存中的数据、寄存器中的值、堆栈中的指令、被打开的文件、各种设备的状态信息等）的总和。

```shell
docker run -it busybox /bin/sh
```

-it参数指的是，在启动容器后，**分配一个文本输入/输出环境，即tty**，与容器的标准输入相关联。

linux的Namespace机制，其实就是Linux新建进行的一个可选参数，在Linux系统中创建线程的系统调用是clone()，如：

```c++
int pid = clone(main_function, stack_size, SIGCHLD, NULL); 
```

这个系统调用会创建一个新进程，并返回其进程号pid（其中，SIGCHLD表示在**一个进程终止或者停止时**，将**SIGCHLD信号发送给其父进程**，按系统默认将忽略此信号，如果父进程希望被告知其子系统的这种状态，则应捕捉此信号）。可以指定CLONE_NEWPID参数，这样就会创建一个**新的PID Namespace**，clone出来的**新进程将成为Namespace里**的**第一个进程**。如：

```c++
int pid = clone(main_function, stack_size, CLONE_NEWPID | SIGCHLD, NULL); 
```

如果多次执行如上的clone()调用，就会创建多个PID Namespace，每个Namespace里的应用进程，都会认为自己是当前容器里的第1号进行，看不到宿主机里真正的进程空间，也看不到其他PID Namespace里的具体情况。

除了PID Namespace，Linux中还有Mount、UTS、IPC、Network和User这些Namespace。

下图左边是虚拟机的工作原理，Hypervisor是虚拟机最主要部分，通过硬件虚拟化功能，模拟出运行一个操作系统需要的各种硬件（CPU、内存、I/O设备等），然后在这些虚拟的硬件上安装了一个新的操作系统Guest OS。所以，这个Hypervisor负责创建虚拟机，会有额外资源消耗和占用，本身虚拟机还会占用内存，对宿主机操作系统的调用也要经过虚拟化软件的拦截和处理，对计算资源、网络和磁盘的I/O损耗也非常大。

右边的Docker在运行时，并没有一个真正的『docker容器』运行在宿主机中，只是一个正常的应用进程，只是在创建时，加上了各种Namespace参数。

『敏捷』和『高性能』是容口相对于虚拟机的最大优势，也是它能够在PaaS这种更细粒度的资源管理平台上大行其道的重要原因。

<html>
<br/>
<img src='../assets/k8s-container-virtual-tech-compare.png' style='max-height: 200px'/>
<br/>
</html>

但基于linux namespace的隔离机制有一个主要问题：**隔离得不彻底**，体现为以下两方面：

+ 容器只是运行在宿主机上的一种特殊进程，多个容器之间用的还是**同一个宿主机的操作系统内核**

可以在容器中通过Mount Namespace单独挂载其他不同版本的操作系统文件，如CentOS或Ubuntu，但不能改变共享宿主机内核的事实。所以，**要在Windows宿主机主运行Linux容器，或者在低版本的Linux宿主机上运行高版本的Linux容器，都是不行的**（Docker on Mac或者windows，实际上都是基于虚拟化技术实现的，和这里要讲的linux容器不同）

+ Linux内核中，**很多资源和对象是不能被Namespace化的**，典型例子就是『时间』

所以，在容器里部署应用时，『什么能做，什么不能做』，是用户必须考虑的。所以容器给应用暴露出的攻击面是很大的，尽管在实践中可以用Seccomp等技术，对容器内部发起的所有系统调用进行过滤和甄别以进行安全加固，但这加多了一层对系统调用的过滤，会拖累容器的性能。所以在生产环境中，不能把运行在物理机上的Linux容器直接暴露到公网上。

注：

> seccomp 是 secure computing 的缩写，其是 Linux kernel 从2.6.23版本引入的一种简洁的 sandboxing 机制。在 Linux 系统里，大量的系统调用（system call）直接暴露给用户态程序。但是，并不是所有的系统调用都被需要，而且不安全的代码滥用系统调用会对系统造成安全威胁。seccomp安全机制能使一个进程进入到一种“安全”运行模式，该模式下的进程只能调用4种系统调用（system call），即 read(), write(), exit() 和 sigreturn()，否则进程便会被终止。

当然，后续讲的**基于虚拟化或者独立内核技术的容器实现**，可以较好地在隔离和性能间做平衡。

## Cgroups

参考[06 \| 白话容器基础（二）：隔离与限制](https://time.geekbang.org/column/article/14653)



## rootfs文件系统

