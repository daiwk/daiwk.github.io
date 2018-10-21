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

[https://blog.csdn.net/weixin_40081916/article/details/79377564](https://blog.csdn.net/weixin_40081916/article/details/79377564)

**auto_ptr**是用于C++11之前的智能指针。由于 auto_ptr 基于排他所有权模式：两个指针不能指向同一个资源，**复制或赋值都会改变资源的所有权。**auto_ptr 主要有两大问题：

+ **复制和赋值会改变资源的所有权**，不符合人的直觉。
+ 在 STL 容器中无法使用auto_ptr ，**因为容器内的元素**必需支持**可复制（copy constructable）**和**可赋值（assignable）**。

unique_ptr特性

+ **拥有**它所指向的**对象**
+ **无法进行复制构造**，也**无法进行复制赋值**操作
+ 保存指向某个对象的指针，当**它本身离开作用域时**会**自动释放它指向的对象**。

unique_ptr可以：

+ 为**动态申请的内存**提供**异常安全**
+ 将动态申请**内存的所有权传递给某个函数**
+ **从某个函数返回动态申请内存的所有权**
+ 在**容器中保存指针**

unique_ptr十分依赖于**右值引用**和**移动语义**。

在C++11中已经**放弃auto_ptr**转而**推荐使用unique_ptr和shared_ptr**。unique跟auto_ptr类似同样只能有一个智能指针对象指向某块内存，但它还有些其他特性。unique_ptr对auto_ptr的改进如下：

+ **auto_ptr支持拷贝构造与赋值操作，但unique_ptr不直接支持**

auto_ptr通过拷贝构造或者operator=赋值后，对象所有权转移到新的auto_ptr中去了，**原来的auto_ptr对象就不再有效**，这点不符合人的直觉。unique_ptr则直接禁止了拷贝构造与赋值操作。

+ **unique_ptr可以用在函数返回值中**

unique_ptr像上面这样一般意义上的复制构造和赋值或出错，但在函数中作为返回值却可以用

+ unique_ptr可做为容器元素

我们知道auto_ptr不可做为容器元素，会导致编译错误。虽然unique_ptr同样**不能直接做为容器元素**，但可以**通过move语义实现**。

unique_ptr持有对对象的独有权，**同一时刻只能有一个**unique_ptr指向给定对象（通过**禁止拷贝**语义、**只有移动**语义来实现）。

unique_ptr指针本身的生命周期：从unique_ptr指针创建时开始，直到离开作用域。

**离开作用域时**，**若其指向对象**，则**将其所指对象销毁**(默认使用delete操作符，用户可指定其他操作)。

[https://blog.csdn.net/qq_33266987/article/details/78784286](https://blog.csdn.net/qq_33266987/article/details/78784286)

### shared_ptr

shared_ptr允许**多个**该智能指针共享地“拥有”同一堆分配对象的内存，这**通过引用计数（reference counting）实现**，会记录有多少个shared_ptr共同指向一个对象，**一旦最后一个这样的指针被销毁**，也就是一旦某个对象的**引用计数变为0**，**这个对象会被自动删除。**

shared_ptr的额外开销：

+ shared_ptr对象除了包括一个所拥有对象的指针外, 还必须包括一个引用计数代理对象的指针.
+ 时间上的开销主要在初始化和拷贝操作上, \*和->操作符重载的开销跟auto_ptr是一样.
+ 开销并不是我们不使用shared_ptr的理由, 永远不要进行不成熟的优化, 直到性能分析器告诉你这一点.

使用：

可以使用模板函数 make_shared 创建对象, make_shared 需指定类型('<>'中)及参数('()'内), 传递的参数必须与指定的类型的构造函数匹配:

```c++
　　std::shared_ptr<int> sp1 = std::make_shared<int>(10);
　　std::shared_ptr<std::string> sp2 = std::make_shared<std::string>("Hello c++");
```

也可以定义 auto 类型的变量来保存 make_shared 的结果。

```c++
　　auto sp3 = std::make_shared<int>(11);
　　printf("sp3=%d\n", *sp3);
　　auto sp4 = std::make_shared<std::string>("C++11");
　　printf("sp4=%s\n", (*sp4).c_str());
```

成员函数：

+ use_count 返回引用计数的个数
+ unique 返回是否是独占所有权( use_count 为 1)
+ swap 交换两个 shared_ptr 对象(即交换所拥有的对象)
+ reset 放弃内部对象的所有权或拥有对象的变更, 会引起原有对象的引用计数的减少
+ get 返回内部对象(指针), 由于已经重载了()方法, 因此和直接使用对象是一样的.如 shared_ptr<int> sp(new int(1)); sp 与 sp.get()是等价的



[https://www.cnblogs.com/lsgxeva/p/7788061.html](https://www.cnblogs.com/lsgxeva/p/7788061.html)

[https://www.cnblogs.com/diysoul/p/5930361.html](https://www.cnblogs.com/diysoul/p/5930361.html)


### weak_ptr

weak_ptr是为配合shared_ptr而引入的一种智能指针来协助shared_ptr工作，它可以从一个shared_ptr或另一个weak_ptr对象构造，**它的构造和析构不会引起引用计数的增加或减少**。没有重载 \* 和 -> 但可以**使用lock获得一个可用的shared_ptr对象**

weak_ptr的使用更为复杂一点，它可以指向shared_ptr指针指向的对象内存，**却并不拥有该内存**，而使用weak_ptr成员lock，则可返回其指向内存的一个share_ptr对象，且**在所指对象内存已经无效时，返回指针空值nullptr**。

小结一下：

+ weak_ptr并不拥有资源的所有权，所以不能直接使用资源。
+ 可以从一个weak_ptr构造一个shared_ptr以取得共享资源的所有权。



## 各种多线程

### atomic

[http://zh.cppreference.com/w/cpp/atomic/atomic](http://zh.cppreference.com/w/cpp/atomic/atomic)


### unique_lock与lock_guard

[https://www.2cto.com/kf/201706/649733.html](https://www.2cto.com/kf/201706/649733.html)

C++多线程编程中通常会对共享的数据进行写保护，以防止多线程在对共享数据成员进行读写时造成资源争抢导致程序出现未定义的行为。通常的做法是在修改共享数据成员的时候进行加锁--mutex。在使用锁的时候通常是在**对共享数据进行修改之前进行lock操作**，在**写完之后再进行unlock操作**，经常会出现由于疏忽，导致由于lock之后，**在离开共享成员操作区域时忘记unlock，导致死锁。**

针对以上的问题，C++11中引入了std::unique_lock与std::lock_guard两种数据结构。通过对lock和unlock进行一次薄的封装，实现**自动unlock**的功能。

