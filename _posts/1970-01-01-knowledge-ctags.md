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
:set tags=/xxx/path/tags(当前路径下的tags文件)
```

或者，设置 ~/.vimrc，加入一行，则不用手动设置tags路径：

```shell
set tags=/xxx/path/tags
```

若要加入系统函数或全局变量的tag标签，则需执行：

```shell
ctags -I __THROW --file-scope=yes --langmap=c:+.h --languages=c,c++ --links=yes --c-kinds=+p --fields=+s -R -f /home/work/.vim/systags /usr/include /usr/local/include /opt/compiler/gcc-4.8.2/include/
```

并且在~/.vimrc中添加（亦可用上面描述的手动加入的方式）：

```shell
set tags+=~/.vim/systags
```

这样，便可以享受系统库函数名补全、原型预览等功能了。


常用命令如下：

+ Ctrl + ]：跳转到变量或函数的定义处，或者用命令
+ Ctrl + o/t：返回到跳转前的位置

