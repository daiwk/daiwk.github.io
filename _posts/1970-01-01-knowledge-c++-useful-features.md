---
layout: post
category: "knowledge"
title: "c++有用的特性"
tags: [c++, ]
---

目录

<!-- TOC -->

- [各种智能指针](#各种智能指针)
    - [unique_ptr](#unique_ptr)
    - [shared_ptr](#shared_ptr)
    - [weak_ptr](#weak_ptr)
- [各种多线程](#各种多线程)
    - [atomic](#atomic)
    - [unique_lock与lock_guard](#unique_lock与lock_guard)

<!-- /TOC -->

## 各种智能指针

### unique_ptr

unique_ptr持有对对象的独有权，同一时刻只能有一个unique_ptr指向给定对象（通过禁止拷贝语义、只有移动语义来实现）。

unique_ptr指针本身的生命周期：从unique_ptr指针创建时开始，直到离开作用域。

离开作用域时，若其指向对象，则将其所指对象销毁(默认使用delete操作符，用户可指定其他操作)。

[https://blog.csdn.net/qq_33266987/article/details/78784286](https://blog.csdn.net/qq_33266987/article/details/78784286)

### shared_ptr

shared_ptr允许多个该智能指针共享第“拥有”同一堆分配对象的内存，这通过引用计数（reference counting）实现，会记录有多少个shared_ptr共同指向一个对象，一旦最后一个这样的指针被销毁，也就是一旦某个对象的引用计数变为0，这个对象会被自动删除。

[https://www.cnblogs.com/lsgxeva/p/7788061.html](https://www.cnblogs.com/lsgxeva/p/7788061.html)

### weak_ptr

weak_ptr是为配合shared_ptr而引入的一种智能指针来协助shared_ptr工作，它可以从一个shared_ptr或另一个weak_ptr对象构造，它的构造和析构不会引起引用计数的增加或减少。没有重载 \* 和 -> 但可以使用lock获得一个可用的shared_ptr对象

weak_ptr的使用更为复杂一点，它可以指向shared_ptr指针指向的对象内存，却并不拥有该内存，而使用weak_ptr成员lock，则可返回其指向内存的一个share_ptr对象，且在所指对象内存已经无效时，返回指针空值nullptr。

注意：

+ weak_ptr并不拥有资源的所有权，所以不能直接使用资源。
+ 可以从一个weak_ptr构造一个shared_ptr以取得共享资源的所有权。

## 各种多线程

### atomic

[http://zh.cppreference.com/w/cpp/atomic/atomic](http://zh.cppreference.com/w/cpp/atomic/atomic)


### unique_lock与lock_guard

[https://www.2cto.com/kf/201706/649733.html](https://www.2cto.com/kf/201706/649733.html)

C++多线程编程中通常会对共享的数据进行写保护，以防止多线程在对共享数据成员进行读写时造成资源争抢导致程序出现未定义的行为。通常的做法是在修改共享数据成员的时候进行加锁--mutex。在使用锁的时候通常是在**对共享数据进行修改之前进行lock操作**，在**写完之后再进行unlock操作**，经常会出现由于疏忽，导致由于lock之后，**在离开共享成员操作区域时忘记unlock，导致死锁。**

针对以上的问题，C++11中引入了std::unique_lock与std::lock_guard两种数据结构。通过对lock和unlock进行一次薄的封装，实现**自动unlock**的功能。

