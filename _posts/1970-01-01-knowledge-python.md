---
layout: post
category: "knowledge"
title: "python小技巧"
tags: [python, ]
---

目录

<!-- TOC -->

- [1. 编译安装python](#1-%E7%BC%96%E8%AF%91%E5%AE%89%E8%A3%85python)
- [2. jupyter](#2-jupyter)
- [3. mkdocs](#3-mkdocs)
- [mac版python3的tf](#mac%E7%89%88python3%E7%9A%84tf)
- [gc](#gc)
      - [引用计数（主要）](#%E5%BC%95%E7%94%A8%E8%AE%A1%E6%95%B0%E4%B8%BB%E8%A6%81)
      - [标记-清除](#%E6%A0%87%E8%AE%B0-%E6%B8%85%E9%99%A4)
      - [分代回收](#%E5%88%86%E4%BB%A3%E5%9B%9E%E6%94%B6)

<!-- /TOC -->

## 1. 编译安装python

python总体上有两个版本，cp27m是ucs2，cp27mu是ucs4，UCS2认为每个字符占用2个字节，UCS4认为每个字节占用4个字符，都是UNICODE的编码形式。

一般基于python的深度学习框架大多默认提供ucs4版本（当然也有ucs2版本），为了以后使用方便，下面我们会用gcc482编译生成python-ucs4版本


1、官网下载python release， https://www.python.org/ftp/python/2.7.14/Python-2.7.14.tgz

2、解压缩到 Python-2.7.14，cd Python-2.7.14

3、编译：

./configure --enable-unicode=ucs4
      打开Makefile，

      修改36行为CC=/opt/compiler/gcc-4.8.2/bin/gcc -pthread

      修改37行为CXX=/opt/compiler/gcc-4.8.2/bin/g++ -pthread

      修改101行(prefix=)为想要编译生成python的位置
      例如 prefix=     /home/work/xxx/exp_space/python-2.7.14

      make

      make install

4、安装pip

      curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

      python get-pip.py

## 2. jupyter

小技巧：如何把整个notebook里的各级目录的东西一起下载？(我们发现在界面里一次只能下载一个文件。。。)

打开一个python窗口，然后输入(google搜出来的啦。。)

```shell
!tar -czhvf notebook.tar.gz *
```

其中的-h参数，是把软链对应的真实文件搞过来哈~。。


## 3. mkdocs

```shell
pip install mkdocs
```

参考：[https://www.mkdocs.org/](https://www.mkdocs.org/)


## mac版python3的tf

在[https://pypi.org/project/tensorflow/1.11.0/#files](https://pypi.org/project/tensorflow/1.11.0/#files)找到tensorflow-1.8.0-cp36-cp36m-macosx_10_11_x86_64.whl这个版本的，下下来(注，如果macosx的版本是10.11以下，好像最高只能装1.8的tf，macosx版本高的话，再试试咯~)

然后

```shell
sudo -H pip3 install ./tensorflow-1.8.0-cp36-cp36m-macosx_10_11_x86_64.whl
```

## gc

参考[https://juejin.im/post/5b34b117f265da59a50b2fbe](https://juejin.im/post/5b34b117f265da59a50b2fbe)

### 引用计数（主要）

在Python中，每一个对象的核心就是一个结构体PyObject，它的内部有一个引用计数器（ob_refcnt）

```c++
typedef struct_object {
 int ob_refcnt;
 struct_typeobject *ob_type;
} PyObject;
```

引用计数的意思就是，一个对象在它刚被New出来的时候因为被New方法引用了所以他的引用计数就是1，如果它被引用（也就是在之前的基础上 例如：b=a，被丢入函数列表等等被引用就会在引用计数上加1），如果引用它的对象被删除的时候（在之前的基础上DEL b）那么它的引用计数就会减少一一直到当它的引用计数变为0的时候，垃圾回收机制就会回收。

优点就是简单快速，缺点是维护成本稍微有点高（多了个refcnt要维护），无法解决循环引用的情况：

```python
a=[1,2]
b=[2,3]
a.append(b)
b.append(a)
DEL a
DEL b
```

### 标记-清除

标记清除就是用来解决循环引用的问题的只有容器对象才会出现引用循环，比如列表、字典、类、元组。
首先，为了追踪容器对象，需要每个容器对象维护两个额外的指针，
用来将容器对象组成一个链表，指针分别指向前后两个容器对象，方便插入和删除操作。试想一下，现在有两种情况：

case1:

```python
a=[1,3]
b=[2,4]
a.append(b)
b.append(a)
del a
del b
```

case2:

```python
a=[1,3]
b=[2,4]
a.append(b)
b.append(a)
del a
```

有两块，一个是root链表(root object)，另外一个是unreachable链表。

对于case1，原来再未执行DEL语句的时候，a,b的引用计数都为2（init+append=2），但是在DEL执行完以后，a,b引用次数互相减1。a,b陷入循环引用的圈子中，然后标记-清除算法开始出来做事，找到其中一端a,开始拆这个a,b的引用环（我们从A出发，因为它有一个对B的引用，则将B的引用计数减1；然后顺着引用达到B，因为B有一个对A的引用，同样将A的引用减1，这样，就完成了循环引用对象间环摘除。），去掉以后发现，a,b循环引用变为了0，所以a,b就被处理到unreachable链表中直接被做掉。

对于case2，简单一看那b取环后引用计数还为1，但是a取环，就为0了。这个时候a已经进入unreachable链表中，已经被判为死刑了，但是这个时候，root链表中有b。如果a被做掉，那世界上还有什么正义... ，在root链表中的b会被进行引用检测引用了a，如果a被做掉了，那么b就...凉凉，一审完事，二审a无罪，所以被拉到了root链表中。

搞两个链表的原因：

之所以要剖成两个链表，是基于这样的一种考虑：现在的unreachable可能存在被root链表中的对象，直接或间接引用的对象，这些对象是不能被回收的，一旦在标记的过程中，发现这样的对象，就将其从unreachable链表中移到root链表中；当完成标记后，unreachable链表中剩下的所有对象就是名副其实的垃圾对象了，接下来的垃圾回收只需限制在unreachable链表中即可。

### 分代回收

了解分类回收，首先要了解一下，GC的阈值，所谓阈值就是一个临界点的值。随着你的程序运行，Python解释器保持对新创建的对象，以及因为引用计数为零而被释放掉的对象的追踪。从理论上说，创建==释放数量应该是这样子。但是如果存在循环引用的话，肯定是创建>释放数量，当创建数与释放数量的差值达到规定的阈值的时候，分代回收机制就登场啦。

垃圾回收=垃圾检测+释放。

分代回收思想将对象分为三代（generation 0,1,2），0代表幼年对象，1代表青年对象，2代表老年对象。根据弱代假说（越年轻的对象越容易死掉，老的对象通常会存活更久。）
新生的对象被放入0代，如果该对象在第0代的一次gc垃圾回收中活了下来，那么它就被放到第1代里面（它就升级了）。如果第1代里面的对象在第1代的一次gc垃圾回收中活了下来，它就被放到第2代里面。```gc.set_threshold(threshold0[,threshold1[,threshold2]])```设置gc每一代垃圾回收所触发的阈值。从上一次第0代gc后，如果分配对象的个数减去释放对象的个数大于threshold0，那么就会对第0代中的对象进行gc垃圾回收检查。 从上一次第1代gc后，如过第0代被gc垃圾回收的次数大于threshold1，那么就会对第1代中的对象进行gc垃圾回收检查。同样，从上一次第2代gc后，如过第1代被gc垃圾回收的次数大于threshold2，那么就会对第2代中的对象进行gc垃圾回收检查。
