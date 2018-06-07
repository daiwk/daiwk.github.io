---
layout: post
category: "knowledge"
title: "编译python"
tags: [python, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

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