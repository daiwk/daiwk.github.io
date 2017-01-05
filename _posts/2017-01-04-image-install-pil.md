---
layout: post
category: "image"
title: "安装PIL"
tags: [paddlepaddle, 集群]
---

```
Libraries have been installed in:
   /home/data/mylib/lib

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

```
echo "/home/data/mylib/lib/" >> /etc/ld.so.conf

ldconfig

cd Imaging-....

python setup.py build_ext -f # force

cp ./build/lib.linux-x86_64-2.7/*.so ./PIL

python selftest.py

[root@cp01-daiwk-docker-test1 Imaging-1.1.7]# python selftest.py
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