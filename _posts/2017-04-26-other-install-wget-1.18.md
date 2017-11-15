---
layout: post
category: "other"
title: "安装wget 1.18+"
tags: [wget 1.18, ]
---

目录

<!-- TOC -->

- [0. 安装](#0-安装)
- [1. 安装gmp](#1-安装gmp)
- [2. 安装nettle](#2-安装nettle)
- [３.　安装gnutl](#３　安装gnutl)
- [4. 安装wget](#4-安装wget)

<!-- /TOC -->

## 0. 安装

## 1. 安装gmp

## 2. 安装nettle


LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib/:/usr/lib/:/usr/lib64/:/opt/lib/:/opt/lib64/ NETTLE_CFLAGS="-I/usr/local/include/" NETTLE_LIBS="-L/usr/local/lib64/ -lnettle" HOGWEED_CFLAGS="-I/usr/local/include" HOGWEED_LIBS="-L/usr/local/lib64/ -lhogweed"


3.1版本的可以，3.3的不行。。产生的makefile没有hogweed..


##　３.　安装gnutl


./configure --with-included-libtasn1=/usr/local/include/ --without-p11-kit

## 4. 安装wget


其实，docker里的1.15就够用了。。
新建一个~/.netrc，内容如下
```
machine machine urs.earthdata.nasa.gov login xxx password xxx
```

然后touch一个空文件urs_cookies

然后执行命令：
```
wget --content-disposition --load-cookies ../.urs_cookies --save-cookies ../.urs_cookies --auth-no-challenge=on --keep-session-cookies -i ../input.urls.txt 
```



