---
layout: post
category: "cv"
title: "安装PIL"
tags: [PIL, jpeg, zlib]
---


目录

<!-- TOC -->

- [0. 一键安装paddle](#0-一键安装paddle)
- [1. 编译jpeg](#1-编译jpeg)
- [2. 安装PIL](#2-安装pil)

<!-- /TOC -->

### 0. 一键安装paddle

如果是度厂内部的机器，记得先装jumbo，然后jumbo装cmake\swig

### 1. 编译jpeg

注意，source ~/.bashrc,保证我们的python是上面安装完的paddle的python

```shell
mkdir ~/mylib
cd ~/mylib

# install jpeglib
export PATH=/opt/compiler/gcc-4.8.2/bin/:$PATH

/bin/rm -rf jpeg*

wget http://www.ijg.org/files/jpegsrc.v8c.tar.gz
tar -xvzf jpegsrc.v8c.tar.gz
LOCAL_PATH=`pwd`
cd jpeg-8c/
./configure --prefix=$LOCAL_PATH --enable-shared --enable-static
make
make install
cd -
```

然后会提示：

```
Libraries have been installed in:
   /home/work/mylib/lib

If you ever happen to want to link against installed libraries
in a given directory, LIBDIR, you must either use libtool, and
specify the full pathname of the library, or use the `-LLIBDIR'
flag during linking and do at least one of the following:
   - add LIBDIR to the `LD_LIBRARY_PATH' environment variable
     during execution
   - add LIBDIR to the `LD_RUN_PATH' environment variable
     during linking
   - use the `-Wl,-rpath -Wl,LIBDIR' linker flag
   - have your system administrator add LIBDIR to `/etc/ld.so.conf'

See any operating system documentation about shared libraries for
more information, such as the ld(1) and ld.so(8) manual pages.
```

于是，我们用root权限：

```
echo "/home/data/mylib/lib/" >> /etc/ld.so.conf
ldconfig
```

或者，没root的话

```
#export LD_LIBRARY_PATH=/home/work/mylib/lib:$LD_LIBRARY_PATH

## jpeglib zlib
export LD_LIBRARY_PATH=/home/work/.jumbo/lib/:/home/work/mylib/lib:$LD_LIBRARY_PATH
```

### 2. 安装PIL

首先，下载安装包并解压

```shell
/bin/rm -rf Imaging*

wget http://effbot.org/downloads/Imaging-1.1.7.tar.gz
tar -xvzf Imaging-1.1.7.tar.gz
cd Imaging-1.1.7/

```

然后，修改setup.py：

```
LOCAL_PATH = "/home/work/mylib/"
JUMBO_PATH = "/home/work/.jumbo/"
JPEG_ROOT = (LOCAL_PATH + "/lib", LOCAL_PATH + "/include")
ZLIB_ROOT = (JUMBO_PATH + "/lib", JUMBO_PATH + "/include")
```

然后，先看看有没有问题

```shell
python setup.py build_ext -f # force

cp ./build/lib.linux-x86_64-2.7/*.so ./PIL

python selftest.py

--------------------------------------------------------------------
PIL 1.1.7 TEST SUMMARY 
--------------------------------------------------------------------
Python modules loaded from ./PIL
Binary modules loaded from ./PIL
--------------------------------------------------------------------
--- PIL CORE support ok
*** TKINTER support not installed
--- JPEG support ok
--- ZLIB (PNG/ZIP) support ok
--- FREETYPE2 support ok
*** LITTLECMS support not installed
--------------------------------------------------------------------
Running selftest:
--- 57 tests passed.

```

没有问题的时候，再

```shell
python setup.py install
```

就可以了！！

