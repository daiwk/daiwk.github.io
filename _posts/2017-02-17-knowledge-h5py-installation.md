---
layout: post
category: "knowledge"
title: "安装h5py"
tags: [h5py]
---

目录

<!-- TOC -->


<!-- /TOC -->

参考：[http://www.cnblogs.com/Ponys/p/3671458.html](http://www.cnblogs.com/Ponys/p/3671458.html)

+ 确定系统有python，numpy，libhdf5-serial-dev，和HDF5.前三者一般都有。这里要安装HDF5

+ 去HDF5官方网站下载编译好的bin（是的，尽管教程让编译，这里给用户的就是编译好的bin，搞得我这小白编译了半天）;

  　　[http://www.hdfgroup.org/HDF5/](http://www.hdfgroup.org/HDF5/)

+ 解压，重命名文件夹为hdf5，移动到 /usr/local/hdf5 下

+ 添加环境变量:

```
　　export HDF5_DIR=/usr/local/hdf5
```

　　到这里HDF5就安装好了，只有安装好的HDF5才能顺利安装h5py

+ pip install h5py


如果在度厂内部机器，如果你已经装了paddle，那么，其实只要

```
jumbo install hdf5
pip install h5py
``` 

就行了。