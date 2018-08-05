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

