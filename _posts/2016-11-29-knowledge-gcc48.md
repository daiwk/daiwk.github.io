---
layout: post
category: "knowledge"
title: "神奇的gcc48"
tags: [gcc48]
---

目录

<!-- TOC -->

- [1.神奇的gcc48](#1神奇的gcc48)
- [2.注意事项](#2注意事项)
    - [1）/opt/compiler/gcc-4.8.2/lib或者lib64添加到LD_LIBRARY_PATH中吗？](#1optcompilergcc-482lib或者lib64添加到ld_library_path中吗)
    - [2）使用gcc4编译得到的so，可以被gcc3编译得到的binary加载吗？](#2使用gcc4编译得到的so可以被gcc3编译得到的binary加载吗)

<!-- /TOC -->

# 1.神奇的gcc48

使用gcc48的最直接方法：

```shell
export PATH=/opt/compiler/gcc-4.8.2/bin:$PATH
```

# 2.注意事项

## 1）/opt/compiler/gcc-4.8.2/lib或者lib64添加到LD_LIBRARY_PATH中吗？

不可以，因为/opt/compiler/gcc-4.8.2/下的库与系统自身的程序和库在OS ABI上不兼容，如果添加之后，系统自身的程序(不要说python了，ls都用不了。。。)将无法运行，会出现如下错误：

```shell
export LD_LIBRARY_PATH=/opt/compiler/gcc-4.8.2/lib64:$LD_LIBRARY_PATH
ls
ls: error while loading shared libraries: /opt/compiler/gcc-4.8.2/lib64/librt.so.1: ELF file OS ABI invalid
```

取巧方法：

```shell
/opt/compiler/gcc-4.8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib:$LD_LIBRARY_PATH /bin/ls
```

## 2）使用gcc4编译得到的so，可以被gcc3编译得到的binary加载吗？ 

不可以，因为gcc3编译得到的binary（譬如系统自带的python）中，其运行时加载的是/lib64/tls/libc.so.6，而gcc4编译得到的so（譬如gcc4编译得到的python扩展sofa.so）可能依赖高版本libc.so的符号。
系统自带的python运行时加载的是系统自身的/lib64/tls/libc.so.6，其libc版本为2.3.4(发现jumbo的python也是一个shi样子……)：

```shell
ldd `which /usr/bin/python`
        libpython2.3.so.1.0 => /usr/lib64/libpython2.3.so.1.0 (0x0000003f0bb00000)
        libpthread.so.0 => /lib64/tls/libpthread.so.0 (0x0000003f0b900000)
        libdl.so.2 => /lib64/libdl.so.2 (0x0000003f0b300000)
        libutil.so.1 => /lib64/libutil.so.1 (0x0000003f0db00000)
        libm.so.6 => /lib64/tls/libm.so.6 (0x0000003f0b500000)
        libc.so.6 => /lib64/tls/libc.so.6 (0x0000003f0b000000)
        /lib64/ld-linux-x86-64.so.2 (0x0000003f0ae00000)

ldd `which ~/.jumbo/bin/python`
        libpython2.7.so.1.0 => /home/users/daiwenkai/.jumbo/lib/libpython2.7.so.1.0 (0x00007ff0d7bea000)
        libpthread.so.0 => /lib64/tls/libpthread.so.0 (0x0000003f0b900000)
        libdl.so.2 => /lib64/libdl.so.2 (0x0000003f0b300000)
        libutil.so.1 => /lib64/libutil.so.1 (0x0000003f0db00000)
        libm.so.6 => /lib64/tls/libm.so.6 (0x0000003f0b500000)
        libc.so.6 => /lib64/tls/libc.so.6 (0x0000003f0b000000)
        /lib64/ld-linux-x86-64.so.2 (0x0000003f0ae00000)
```

而gcc4编译得到的python扩展_CRFPP.so依赖高版本libc.so的符号：

```shell
nm ~/CRF_LIB/_CRFPP.so | grep GLIBC
                 U _ZdaPv@@GLIBCXX_3.4
                 U _Znam@@GLIBCXX_3.4
                 w __cxa_finalize@@GLIBC_2.2.5
                 U fputc@@GLIBC_2.2.5
                 U fputs@@GLIBC_2.2.5
                 U free@@GLIBC_2.2.5
                 U fwrite@@GLIBC_2.2.5
                 U malloc@@GLIBC_2.2.5
                 U memcpy@@GLIBC_2.14
                 U printf@@GLIBC_2.2.5
                 U strcmp@@GLIBC_2.2.5
                 U strlen@@GLIBC_2.2.5
                 U strncmp@@GLIBC_2.2.5
                 U strncpy@@GLIBC_2.2.5
                 U strstr@@GLIBC_2.2.5
```

参考：

[http://unix.stackexchange.com/questions/132158/how-do-i-maintain-a-separate-newer-glibc-gcc-stack-as-non-root-on-linu
](http://unix.stackexchange.com/questions/132158/how-do-i-maintain-a-separate-newer-glibc-gcc-stack-as-non-root-on-linu)
[http://stackoverflow.com/questions/847179/multiple-glibc-libraries-on-a-single-host](http://stackoverflow.com/questions/847179/multiple-glibc-libraries-on-a-single-host)
rtldi: [http://bitwagon.com/rtldi/rtldi.html](http://bitwagon.com/rtldi/rtldi.html)


所以呢，就悲剧了，有一个很trick的办法，可以让gcc48编译出来的so给gcc34编译出来的py用：

```shell
/opt/compiler/gcc-4.8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib `which python` -c "import CRFPP"
```

当然了，这有点太恶心了吧。。。我们发现，其实直接用gcc34也可以编译出CRFPP的。。那么我们就直接用就好啦：

```shell

./configure
make
cd ./python/
python setup.py build
python setup.py install

export LD_LIBRARY_PATH=~/crfpp/CRF++-0.58.gcc3.4.5/.libs/:$LD_LIBRARY_PATH 
python -c "import CRFPP"

nm ./build/lib.linux-x86_64-2.7/_CRFPP.so | grep GLIBC
                 w __cxa_finalize@@GLIBC_2.2.5
                 U fputs@@GLIBC_2.2.5
                 U free@@GLIBC_2.2.5
                 U malloc@@GLIBC_2.2.5
                 U memcpy@@GLIBC_2.2.5
                 U memset@@GLIBC_2.2.5
                 U printf@@GLIBC_2.2.5
                 U strcmp@@GLIBC_2.2.5
                 U strcpy@@GLIBC_2.2.5
                 U strlen@@GLIBC_2.2.5
                 U strncmp@@GLIBC_2.2.5
                 U strncpy@@GLIBC_2.2.5
                 U strstr@@GLIBC_2.2.5
                 U _ZdaPv@@GLIBCXX_3.4
                 U _Znam@@GLIBCXX_3.4

```

完美……