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
            - [wait-free write](#wait-free-write)
        - [bthread_id](#bthread_id)
        - [rpc::LoadBalance](#rpcloadbalance)
            - [调度并发](#调度并发)
            - [locality aware](#locality-aware)
            - [consistent hashing](#consistent-hashing)
        - [rpc::NamingService](#rpcnamingservice)
    - [rpc::Channel](#rpcchannel)
        - [支持协议：](#支持协议)
        - [特性：](#特性)
    - [rpc::Server](#rpcserver)
        - [同端口多协议](#同端口多协议)
        - [特性](#特性)
    - [内置服务](#内置服务)
        - [/status](#status)
        - [/connections](#connections)
        - [/flags](#flags)
        - [/vars](#vars)
        - [/rpcz](#rpcz)
        - [/pprof/profiler](#pprofprofiler)
        - [/pprof/heap](#pprofheap)
    - [baidurpc的性能](#baidurpc的性能)
    - [其他语言实现](#其他语言实现)
    - [bthread](#bthread)
        - [eventloop](#eventloop)
        - [eventloop+threadpool](#eventloopthreadpool)
        - [bthread](#bthread-1)
    - [bvar](#bvar)
    - [base](#base)

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

这里的fd指的是socket

+ 使fd ABA-free：
linux管理fd时用的是32位的整数，但关掉之后就释放了，可能被另一个进程打开，而指向了另一个设备。所以多线程时，可能同一个整数被多个线程使用，用的时候可能fd已经被关掉或者被重复打开（ABA-连续看到两个值是a，无法判断是之前那个a，还是开始是a，后来变b，又变回a）
+ 能原子地写入消息：
在操作系统中，fd是可以原子地写入消息的，但这里的原子指的是byte。例如，有两个线程要往一个fd写消息，但一个消息的长度肯定不止一个byte，所以结果会是两个线程交错地写这个fd。

##### wait-free write

使每个线程同时能做有用的事，而不是因为锁被一个线程抢占了，其他线程就只能空等(one-free)，而如果用了锁，就两者都不能保证。每秒可以写入500w个16字节的消息。
解决：使用一个64-bid的id。多线程的时候，不能传指针之类的，因为不知道什么时候就被free了，可能访问的是一个非法内存。所以用一个64位的id，并且被一个类似sharedptr的包住，保证不会在中间被析构，当有一个地方将其标记为失效时，其他用到的地方也都会原子地失效

#### bthread_id

+ ABA-free: 同上，是个64bit的id

其他rpc存在的问题：
例1：当往一个连接开始写一个request之后，可能response在write这个函数完成之前就回来了。而如果write的时候会访问一个数据，而拿到response后也会访问同一个数据，那就会有问题
例2：当超时时间设置得很短，线程还在写连接的时候，超时时间就到了，而连接里的东西不完整，可能别人会去读它。

+ checking every RPC call without global contention(??)

#### rpc::LoadBalance

详见[https://github.com/brpc/brpc/blob/master/docs/cn/lalb.md#doublybuffereddata](https://github.com/brpc/brpc/blob/master/docs/cn/lalb.md#doublybuffereddata)

例如，round-robin（rr）。

##### 调度并发

例如对于rr算法而言，因为naming service对应的下游可能会变，所以每个线程在访问rr的这个列表时，常规的解决方法之一就是用读写锁。例如在查时，用的是读锁，而想要修改时，会用写锁。但在POSIX中，读写锁的性能会特别差。如果临界区不是特别大（例如rr其实临界区非常小），会发现直接用mutex反而比读写锁还要快…

brpc的解决方法：用一个特殊的双buffer读写锁，读时读前台的buffer，写时写后台的buffer，定时同步两个buffer（前后台切换）。但有一个限制就是前后台不能切换得太快，不然可能出现类似前面讲到的ABA的情况，可能一个线程正在读前台，但中间后台和前台切换了，然后又切换一次。所以会有一个切换频率，最低也要2s，一般是5-10s。但在loadbalance这个场景，这个等待时间是不可接受的。所以这里是一个特殊的双buffer读写锁，只需要和每一个前台线程抢一把thread-local锁就行了。

##### locality aware

最适合混部的分流算法。是一个动态的迭代型的算法，总是能选择期望延时最低的server。例如，优先把流量分给同机房的机器，只有当同机房的机器或者负载达到一定临界值，或者出现故障的时候，才会导流给邻近机房的机器。

##### consistent hashing

实现了多种一致性hash，便于各类caching使用。主要是两种：
+ 基于memcached的
+ 基于md5的

#### rpc::NamingService

+ 统一形式： protocol://url
+ 例如：bns://node, http://node, file://path, list://server1,server2,..., servername

### rpc::Channel

与服务器通信

#### 支持协议：
+ hulu-pbrpc,sofa-pbrpc,public/pbrpc,nova-pbrpc,ubrpc(idl/mcpack/compack)
+ http 1.1, hadoop-rpc, memcached

#### 特性：
+ 线程安全，都支持异步
也就是说，不需要每个线程都建一个channel。很多开源的都是线程不安全的

+ 超时，重试，backup request， 取消
    + 超时是严格的超时，只要超时就一定结束，结束就一定失败，错误码是timeout。而其他rpc，可能有很多种不同的超时。
    + 重试只会发生在连接断开时，而不会在各种超时的时候去不断开连接继续重试（这种情况就是backup request了）
    + backup request就是比如设置了5ms没回来，就再try一下，再等5ms
    + 取消，可以随时取消。如果是异步调用，取消后仍然会调用done，返回的错误码是cancel

+ 单连接，连接池，短连接，连接caching，连接认证
    + 单连接，不管建立多少个channel，两点之前都只会建立一个连接。所以就会使用前面讲到的wait-free的fd的socket。
    + 连接池，ub、http1.1（keep-alive）、public/pbrpc就是基于这种方式的，相当于一个池子里有n个连接，一个请求想发送时，就从这n个连接里拿一个来用，用完就放回池子里，并不需要大量地重新建立连接。但，如果在一个集群里，如果两两间要建立连接，而单点和单点之间假设要保持很多个连接，那fd、端口号可能就会被打爆了，所以就需要单连接的设计
    + 短连接，用完就释放掉，适用于频率非常低的，比如一秒来一次的
    + 连接caching，server端可以配一个参数，例如某个连接大于5s，没有任何东西写入，就把连接关掉。client端也有
    + 连接认证，支持认证
+ ParallelChannel：更简单的并发访问方式。是一个combo-channel，可以加入一系列的子channel，每个Channel对应一个子分库，这样，对一个parallellchannel的访问，就会变成对这一系列子channel的并发访问（全异步）。可以parallelchannel套parallelchannel

channel的析构是不影响这次rpc_call的。所以如果是点对点的channel，那可以在栈上直接声明一个channel变量，然后去异步地rpc_call，再做别的。但如果是在bns上的channel，因为本身这个init是比较重量级的，就不太好这么用，最好还是用一个类成员变量之类的。

```c++
rpc::Channel xx;
xx.Init("10.1.1.1", "la", NULL);
```

### rpc::Server

#### 同端口多协议

+ hulu-pbrpc,sofa-pbrpc,public/pbrpc,nova-pbrpc,ubrpc(with adapter,将idl适配成pb，server中还是pb，但client可以仍然用idl)
+ http 1.1, hadoop-rpc, https

#### 特性

+ 高度并发：只要能并发就并发。除了从一个fd读数据之外，因为操作系统要求读一个fd只能是一个线程读，其实是一个线程不安全的设计。例如一个fd里传来了两个pb，如果pb很大，普通的rpc会等第一个parsefrom完再去解第二个，而brpc可以并发解析。
+ 没有io线程和worker线程之分，首先，是没有io线程的，只有worker线程，其次，默认取cpu的核数，会在/flags里记录，可以通过内置的html页面去动态地修改
+ 同进程内所有Server/Channel默认共享工作线程。为了提高线程的利用率

其中，```SERVER_OWNS_SERVICE```指的是server析构的时候，一起把service给干掉；反之，```SERVER_DOESNT_OWN_SERVICE```指的是service析构时，不把service干掉

```c++
rpc::Server xx;
xx.AddService(new MyService(), rpc::SERVER_OWNS_SERVICE);
xx.Start(8010, NULL);
```

### 内置服务

方便监控和调试

#### /status

因为同端口支持多协议，所以可以用同一个server端口在浏览器打开。可以看到支持的所有service,还有每个method的对应的指标，各种时间维度的流量、平响之类的

#### /connections

精确到微秒，有各种对端的ip之类的

#### /flags

进程内所有的gflags，有（R）的，就表示可以在浏览器里动态修改，改过的会高亮。如果一个gflag有检查函数，就会动态reload(gflag的特性)

#### /vars

所有用到bvar的都会显示在这里，bvar可以算各种cnt,max,min,90perlatency,80perlatency之类的，类似ubmonitor，但性能好很多，会定时写，noah会动态地读

#### /rpcz

会把进程内所有rpc_call列出来，会分配一个唯一的traceid，会有每一次rpc_call的详情

#### /pprof/profiler

支持远程做profiling，可以在线做cpu /heap profiling，

#### /pprof/heap

同上

### baidurpc的性能

好于ub/hulu/sofa/thrift/zeromq...

why：

+ 从sys_read开始就是并发的，只要cpu有富余，请求总会在O(1)时间内开始处理
+ wait-free write 写出总会在O(1)时间内返回，高吞吐，特别是包大的时候
+ 高并发的loadbalancer（因为locality aware比较复杂，所以要求localbalancer不能全局加锁）
+ 没有全局竞争的request tracking(不需要一个全局hash表来区分每次请求，有bthread-id就行了)
+ 不区分io与worker，可以减少一次上下文切换
+ dedicated memory allocator: 专门写的针对多线程的内存分配，比tcmalloc更快

### 其他语言实现

+ python: 
    + 基于c++包装，通过动态pb互动
    + 尽量规避GIL(GIL并不是Python的特性，它是在实现Python解析器(CPython)时所引入的一个概念。参考[http://python.jobbole.com/81822/](http://python.jobbole.com/81822/))
+ java：完全用java写的
+ 其他脚本语言：推荐用http+json访问baidurpc，因为baidurpc默认开启http

### bthread

M:N的线程库，同步的代码可以获得异步的性能，和pthread接口同构

butex使bthread阻塞函数可同时被bthread和pthread调用，分别阻塞bthread和pthread，相互可唤醒。mutex/semaphore/condition之类的，都是基于futex（POSIX最底层的）的，而butex就相当于futex。

超快的创建：每个rpc请求建一个bthread,请求结束，bthread就结束。如果一个channel里有3个请求，会用一个bthread读进来，然后建两个bthread去处理后面两个请求，读数据的那个bthread处理第一个请求

超快的scheduler
+ 更好的cache locality: 允许新线程在当前cpu core上执行
+ 减少全局竞争：取线程任务靠steal，而非pull

#### eventloop

从epoll-wait开始，后面的callback会等上面的结束了再执行，会进到epoll-wait状态，延时不可控，有的callback很慢，所以会等很久。

适用于高度定制的

ub_aserver用的就是eventloop，有n个eventloop。

#### eventloop+threadpool

io线程+worker线程

缺点：
+ io线程的竞争非常激烈
+ io线程到worker线程有拷贝的开销
+ epoll-control在linux中的实现，时间复杂度要O(n)

#### bthread

需要执行callback时，复用io线程

### bvar

修改是thread-local的，写数据的时候并不急着读，所以写自己的thread-local就行，不需要全局竞争，只在需要读的时候汇总就行了

### base

基于chromium【[https://chromium.googlesource.com/chromium/src/+/master/docs/linux_build_instructions.md](https://chromium.googlesource.com/chromium/src/+/master/docs/linux_build_instructions.md)】和百度的一些公共库写的
