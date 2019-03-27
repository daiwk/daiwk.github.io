---
layout: post
category: "knowledge"
title: "linux"
tags: [linux, ]
---

目录

<!-- TOC -->

- [查看机器支持的指定集](#查看机器支持的指定集)
- [每一条命令都显示执行的时间戳](#每一条命令都显示执行的时间戳)

<!-- /TOC -->

## 查看机器支持的指定集

```shell
work@xxx$ cat /proc/cpuinfo | grep flags | uniq
flags           : fpu vme de pse tsc msr pae mce cx8 apic mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 ds_cpl vmx smx est tm2 ssse3 fma cx16 xtpr pdcm dca sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch arat epb xsaveopt pln pts dts tpr_shadow vnmi flexpriority ept vpid fsgsbase bmi1 hle avx2 smep bmi2 erms invpcid rtm
```

## 每一条命令都显示执行的时间戳

首先要安装ts命令，参考[https://rentes.github.io/unix/utilities/2015/07/27/moreutils-package/](https://rentes.github.io/unix/utilities/2015/07/27/moreutils-package/)。如果在centos，就：

```shell
yum install moreutils
```

然后假设sh文件是：

```shell
set -x
echo "a"
echo "b"
echo "c"
echo "d"
```

于是我们执行：

```shell
sh xx.sh 2>&1 | ts '[%Y-%m-%d %H:%M:%S]'
```

就可以看到：

```shell
[2019-03-27 11:10:34] + echo a
[2019-03-27 11:10:34] a
[2019-03-27 11:10:34] + echo b
[2019-03-27 11:10:34] b
[2019-03-27 11:10:34] + echo c
[2019-03-27 11:10:34] c
[2019-03-27 11:10:34] + echo d
[2019-03-27 11:10:34] d
```