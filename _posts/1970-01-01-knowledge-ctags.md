---
layout: post
category: "knowledge"
title: "ctags"
tags: [ctags, ]
---

目录

<!-- TOC -->

- [ctags](#ctags)

<!-- /TOC -->

## ctags

参考[https://www.cnblogs.com/zl-graduate/p/5777711.html](https://www.cnblogs.com/zl-graduate/p/5777711.html)


首先，执行

```shell
ctags -R ./* 
```

【!!!注意：不是```ctags ./* -R```!!!】

在vim打开源码时，指定tags文件，才可正常使用，通常手动指定，在vim命令行输入：

```shell
:set tags=./tags(当前路径下的tags文件)
```

常用命令如下：

+ Ctrl + ]：跳转到变量或函数的定义处，或者用命令
+ Ctrl + o/t：返回到跳转前的位置

