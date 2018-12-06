---
layout: post
category: "knowledge"
title: "c++有用的特性"
tags: [c++, ]
---

目录

<!-- TOC -->

- [字符串相关](#%E5%AD%97%E7%AC%A6%E4%B8%B2%E7%9B%B8%E5%85%B3)
    - [缓冲区溢出问题](#%E7%BC%93%E5%86%B2%E5%8C%BA%E6%BA%A2%E5%87%BA%E9%97%AE%E9%A2%98)
        - [strncpy/strncat](#strncpystrncat)
        - [snprintf](#snprintf)
- [各种容器](#%E5%90%84%E7%A7%8D%E5%AE%B9%E5%99%A8)
    - [map与unordered map对比](#map%E4%B8%8Eunordered-map%E5%AF%B9%E6%AF%94)
- [各种智能指针](#%E5%90%84%E7%A7%8D%E6%99%BA%E8%83%BD%E6%8C%87%E9%92%88)
    - [unique_ptr](#uniqueptr)
    - [shared_ptr](#sharedptr)
    - [weak_ptr](#weakptr)
- [各种多线程](#%E5%90%84%E7%A7%8D%E5%A4%9A%E7%BA%BF%E7%A8%8B)
    - [thread基本用法](#thread%E5%9F%BA%E6%9C%AC%E7%94%A8%E6%B3%95)
    - [thread_local](#threadlocal)
    - [atomic](#atomic)
    - [unique_lock与lock_guard](#uniquelock%E4%B8%8Elockguard)
- [其他](#%E5%85%B6%E4%BB%96)
    - [值/引用/指针](#%E5%80%BC%E5%BC%95%E7%94%A8%E6%8C%87%E9%92%88)

<!-- /TOC -->

## 字符串相关

### 缓冲区溢出问题

[https://www.cnblogs.com/clover-toeic/p/3737011.html](https://www.cnblogs.com/clover-toeic/p/3737011.html)

#### strncpy/strncat

该对函数是strcpy/strcat调用的“安全”版本，但仍存在一些问题：

+ strncpy和strncat要求程序员给出剩余的空间，而不是给出缓冲区的总大小。缓冲区大小一经分配就不再变化，但缓冲区中剩余的空间量会在每次添加或删除数据时发生变化。这意味着程序员需始终跟踪或重新计算剩余的空间，而这种跟踪或重新计算很容易出错。
+ 在发生溢出(和数据丢失)时，strncpy和strncat返回结果字符串的起始地址(而不是其长度)。虽然这有利于链式表达，但却无法报告缓冲区溢出。
+ 若源字符串长度至少和目标缓冲区相同，则strncpy不会使用NUL来结束字符串；这可能会在以后导致严重破坏。因此，在执行strncpy后通常需要手工终止目标字符串。
+ strncpy还可复制源字符串的一部分到目标缓冲区，要复制的字符数目通常基于源字符串的相关信息来计算。这种操作也会产生未终止字符串。
+ strncpy会在源字符串结束时使用NUL来填充整个目标缓冲区，这在源字符串较短时存在性能问题。

#### snprintf

sprintf使用控制字符串来指定输出格式，该字符串通常包括"%s"(字符串输出)。若指定字符串输出的精确指定符，则可通过指定输出的最大长度来防止缓冲区溢出(如%.10s将复制不超过10个字符)。也可以使用"*"作为精确指定符(如"%.*s")，这样就可传入一个最大长度值。精确字段仅指定一个参数的最大长度，但缓冲区需要针对组合起来的数据的最大尺寸调整大小。

"字段宽度"(如"%10s"，无点号)仅指定最小长度——而非最大长度，从而留下缓冲区溢出隐患。

参考[https://www.cnblogs.com/52php/p/5724390.html](https://www.cnblogs.com/52php/p/5724390.html)

```c++
int snprintf(char *restrict buf, size_t n, const char * restrict  format, ...);
```

函数说明：**最多**从源串中拷贝**n－1**个字符到目标串中，然后**再在后面加一个 ```'\0'```**。所以如果**目标串的大小为 n 的话，将不会溢出**。

函数返回值：若成功则返回欲写入的字符串长度(而不是实际写入的字符串度)，若出错则返回负值。

用法：

```c++
    char str[10] = {0,};
    snprintf(str, sizeof(str), "0123456789012345678");
    printf("str=%s\n", str);
```

## 各种容器

### map与unordered map对比

[https://blog.csdn.net/ljp1919/article/details/50463761](https://blog.csdn.net/ljp1919/article/details/50463761)

效率上：

+ boost::unordered_map （34s）插入比map(49s)快。
+ boost::unordered_map （15s）查找操作比map(23s)快。

内存空间占用上：

+ boost::unordered_map 内存占用26%。7.6GB*0.26=1.976GB。
+ map内存占用29%。7.6GB*0.29=2.2GB。

所以，**在效率和内存占用上面，boost::unordered_map 都更具有优势**。


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

类似shared_ptr，用unique_ptr管理非new对象、没有析构函数的类时，需要向unique_ptr传递一个删除器。不同的是，unique_ptr管理删除器的方式，我们必须在尖括号中unique_ptr指向类型后面提供删除器的类型，在创建或reset一个这种unique_ptr对象时，必须提供一个相同类型的可调用对象（删除器），这个删除器接受一个T*参数。

unique_ptr不允许两个独占指针指向同一个对象，在没有裸指针的情况下，我们只能用release获取内存的地址，同时放弃对对象的所有权，这样就有效避免了多个独占指针同时指向一个对象。 而使用裸指针就很容易打破这一点。

在调用u.release()时是不会释放u所指的内存的，这时返回值就是对这块内存的唯一索引，如果没有使用这个返回值释放内存或是保存起来，这块内存就泄漏了。


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

### thread基本用法

[https://www.cnblogs.com/wangguchangqing/p/6134635.html](https://www.cnblogs.com/wangguchangqing/p/6134635.html)

### thread_local

参考[https://www.cnblogs.com/pop-lar/p/5123014.html](https://www.cnblogs.com/pop-lar/p/5123014.html)

thread_local变量是C++ 11新引入的一种存储类型。它会影响变量的存储周期(Storage duration)，C++中有4种存储周期：

+ automatic
+ static
+ dynamic
+ thread

有且只有thread_local关键字修饰的变量具有线程周期(thread duration)，这些变量(或者说对象）在线程开始的时候被生成(allocated)，在线程结束的时候被销毁(deallocated)。并且每 一个线程都拥有一个独立的变量实例(Each thread has its own instance of the object)。thread_local 可以和static 与 extern关键字联合使用，这将影响变量的链接属性(to adjust linkage)。

那么，哪些变量可以被声明为thread_local？以下3类都是ok的

+ 命名空间下的全局变量
+ 类的static成员变量
+ 本地变量

既然每个线程都拥有一份独立的thread_local变量，那么就有2个问题需要考虑：

+ 各线程的thread_local变量是如何初始化的
+ 各线程的thread_local变量在初始化之后拥有怎样的生命周期，特别是被声明为thread_local的本地变量(local variables)

### atomic

[http://zh.cppreference.com/w/cpp/atomic/atomic](http://zh.cppreference.com/w/cpp/atomic/atomic)


### unique_lock与lock_guard

[https://www.2cto.com/kf/201706/649733.html](https://www.2cto.com/kf/201706/649733.html)

C++多线程编程中通常会对共享的数据进行写保护，以防止多线程在对共享数据成员进行读写时造成资源争抢导致程序出现未定义的行为。通常的做法是在修改共享数据成员的时候进行加锁--mutex。在使用锁的时候通常是在**对共享数据进行修改之前进行lock操作**，在**写完之后再进行unlock操作**，经常会出现由于疏忽，导致由于lock之后，**在离开共享成员操作区域时忘记unlock，导致死锁。**

针对以上的问题，C++11中引入了std::unique_lock与std::lock_guard两种数据结构。通过对lock和unlock进行一次薄的封装，实现**自动unlock**的功能。

## 其他

### 值/引用/指针

[https://segmentfault.com/a/1190000006812825](https://segmentfault.com/a/1190000006812825)
